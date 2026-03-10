"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Camera, Play, Square, Eye, Zap, Radio, Target } from "lucide-react";
import { logger } from "@/lib/logger";

// Type definitions for the data we get from the backend
interface Detection {
  class_name: string;      // e.g., "person", "car", "dog"
  confidence: number;       // How sure the AI is (0.0 to 1.0)
  priority: number;         // Safety priority (10 = stop now, 1 = ignore)
  category: string;         // Object category (living, vehicle, obstacle)
  centroid: number[];       // Center point [x, y] for navigation
}

interface SafetyAlert {
  object: string;
  position: string;         // "left", "center", or "right"
  distance: string;         // "close", "medium", "far"
  urgency: "critical" | "high" | "medium" | "low";
  action_required: boolean; // Does the robot need to do something?
}

interface NavigationRecommendation {
  action: "proceed" | "maneuver" | "stop";
  direction: string;        // "forward", "left", "right", "none"
  risk_level: "critical" | "high" | "medium" | "low" | "safe";
  total_objects_detected: number;
  safe_directions: { left: boolean; center: boolean; right: boolean };
}

interface SegmentationResult {
  inference_time_ms: number;  // How long the AI took to process
  fps: number;                // Frames per second
  object_count: number;       // How many objects detected
  detections: Detection[];
  safety_alerts: SafetyAlert[];
  navigation_recommendation: NavigationRecommendation;
}

// Color scheme for the retro terminal UI
// Priority 10 (people) = red, priority 3 (chairs) = gray, etc.
const priorityColors: Record<number, string> = {
  10: "text-red-400",
  9: "text-orange-400",
  8: "text-yellow-400",
  7: "text-cyan-400",
  5: "text-green-400",
  3: "text-gray-400",
};

const riskColors: Record<string, string> = {
  critical: "text-red-500",
  high: "text-orange-500",
  medium: "text-yellow-500",
  low: "text-green-500",
  safe: "text-emerald-500",
};

export default function SegmentationPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
  const [framesReceived, setFramesReceived] = useState(0);
  const startTimeRef = useRef<number>(Date.now());

  // Check service status
  const checkServiceStatus = useCallback(async () => {
    try {
      const response = await fetch("/api/segment");
      if (response.ok) {
        const data = await response.json();
        setIsConnected(true);
        setModelLoaded(data.model?.loaded ?? false);
      }
    } catch {
      setIsConnected(false);
      setModelLoaded(null);
    }
  }, []);

  useEffect(() => {
    const checkStatus = async () => { await checkServiceStatus(); };
    void checkStatus();
    const interval = setInterval(() => void checkServiceStatus(), 5000);
    return () => clearInterval(interval);
  }, [checkServiceStatus]);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const overlayEnabledRef = useRef(showOverlay);

  useEffect(() => {
    overlayEnabledRef.current = showOverlay;
  }, [showOverlay]);

  // Set up WebSocket connection to the Python backend
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Use environment variable if set, otherwise default to local development
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:3030/";
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.info("WebSocket connected to segmentation service");
      setIsConnected(true);  // Update UI to show "connected"
    };

    ws.onmessage = (event) => {
      try {
        const response = JSON.parse(event.data);
        if (response.type === "result") {
          // Got detection results from backend
          setResult(response.data);
          setResultImage(response.result_image || null);
          setError(null);
          setFramesReceived(prev => prev + 1);  // For debugging/display
          readyForNextFrame.current = true;  // Tell the sender we can send another frame
        } else if (response.type === "error") {
          logger.error("Segmentation error:", response.error);
          readyForNextFrame.current = true;
        }
      } catch {
        logger.error("WebSocket message parse error");
        readyForNextFrame.current = true;
      }
    };

    ws.onerror = () => {
      logger.error("WebSocket error occurred");
      setIsConnected(false);
    };

    ws.onclose = () => {
      logger.info("WebSocket disconnected");
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, []);

  // Back-pressure control: only send a new frame after the server processed the last one
  // This prevents lag build-up and keeps latency low
  const readyForNextFrame = useRef(true);

  // Capture current video frame and send to backend for processing
  const sendFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!canvas || !video || !ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Resize to 640x480 - this is what YOLO expects
    const targetWidth = 640;
    const targetHeight = 480;
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, targetWidth, targetHeight);

    // Convert to base64 JPEG (70% quality for faster transmission)
    const imageData = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    const overlay = overlayEnabledRef.current;

    // Send frame to backend
    ws.send(JSON.stringify({
      type: "frame",
      image: imageData,
      overlay,
    }));

    readyForNextFrame.current = false;  // Wait for server response before sending more
  }, []);

  // Start the frame sending loop - checks every 33ms (~30 FPS) but only sends when server is ready
  const startFrameSending = useCallback(() => {
    const FRAME_INTERVAL = 33; // Check at ~30 Hz

    frameIntervalRef.current = setInterval(() => {
      if (readyForNextFrame.current) {
        sendFrame();
      }
    }, FRAME_INTERVAL);
  }, [sendFrame]);

  // Start camera
  const startStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      connectWebSocket();
      // Small delay to ensure WS connects before sending frames
      setTimeout(() => startFrameSending(), 500);
    } catch {
      setError("Camera access denied");
    }
  }, [connectWebSocket, startFrameSending]);

  // Stop camera
  const stopStream = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsStreaming(false);
    setResultImage(null);
    setResult(null);
  }, []);

  useEffect(() => {
    return () => { stopStream(); };
  }, [stopStream]);

  // Format uptime
  const uptime = Math.floor((Date.now() - startTimeRef.current) / 1000);
  const hours = Math.floor(uptime / 3600);
  const mins = Math.floor((uptime % 3600) / 60);
  const secs = uptime % 60;

  return (
    <div className="min-h-screen bg-black text-green-400 font-mono p-4">
      {/* Header */}
      <header className="border border-green-800 p-3 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-2xl font-bold tracking-wider">mybot_vision</div>
            <span className="text-green-600 text-xs">v2.0.1</span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span>UPTIME: {hours.toString().padStart(2, "0")}:{mins.toString().padStart(2, "0")}:{secs.toString().padStart(2, "0")}</span>
            <span className={isConnected ? "text-green-400" : "text-red-400"}>
              ● {isConnected ? "LINK_OK" : "NO_LINK"}
            </span>
            <span className={modelLoaded ? "text-cyan-400" : "text-yellow-400"}>
              ■ {modelLoaded ? "MODEL_READY" : "NO_MODEL"}
            </span>
            <span className="text-green-500">
              ▣ FRAMES: {framesReceived}
            </span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Video Feed - Takes 3 columns */}
        <div className="lg:col-span-3">
          <div className="border border-green-800 p-2">
            <div className="flex items-center justify-between mb-2 text-xs text-green-600 border-b border-green-900 pb-1">
              <span>&gt; VIDEO_FEED</span>
              <div className="flex gap-4">
                <span>FPS: {result?.fps.toFixed(1) || "0.0"}</span>
                <span>LATENCY: {result?.inference_time_ms.toFixed(0) || "0"}ms</span>
              </div>
            </div>

            {/* Video Container */}
            <div className="relative aspect-video bg-gray-950 border border-green-900">
              <video
                ref={videoRef}
                className={`absolute inset-0 w-full h-full object-cover ${showOverlay && resultImage ? "hidden" : ""}`}
                playsInline
                muted
              />

              {showOverlay && resultImage && (
                <img
                  src={`data:image/jpeg;base64,${resultImage}`}
                  alt="Segmentation"
                  className="absolute inset-0 w-full h-full object-cover"
                />
              )}

              <canvas ref={canvasRef} className="hidden" />

              {!isStreaming && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="h-12 w-12 mx-auto mb-2 text-green-800" />
                    <span className="text-green-600 text-sm">
                      {isConnected ? "[ CLICK START TO BEGIN ]" : "[ WAITING FOR SERVICE ]"}
                    </span>
                  </div>
                </div>
              )}

              {/* Navigation HUD */}
              {result && isStreaming && (
                <div className="absolute bottom-2 left-2 right-2 border border-green-700 bg-black/80 p-2">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className={riskColors[result.navigation_recommendation.risk_level]}>
                        ▸ {result.navigation_recommendation.action.toUpperCase()}
                      </span>
                      {result.navigation_recommendation.direction !== "none" && (
                        <span className="text-green-300">
                          [{result.navigation_recommendation.direction.toUpperCase()}]
                        </span>
                      )}
                    </div>
                    <div className="flex gap-1">
                      {(["left", "center", "right"] as const).map((dir) => (
                        <span
                          key={dir}
                          className={`px-2 py-0.5 border ${result.navigation_recommendation.safe_directions[dir]
                            ? "border-green-500 text-green-400"
                            : "border-red-500 text-red-400"
                            }`}
                        >
                          {dir[0].toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Error overlay */}
              {error && (
                <div className="absolute top-2 left-2 right-2 border border-red-800 bg-red-950/80 p-2 text-red-400 text-xs">
                  ⚠ {error}
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-green-900">
              <div className="flex gap-2">
                {!isStreaming ? (
                  <Button
                    onClick={startStream}
                    disabled={!isConnected || modelLoaded === false}
                    className="bg-green-900 hover:bg-green-800 text-green-400 border border-green-700 font-mono"
                  >
                    <Play className="h-4 w-4 mr-1" /> START
                  </Button>
                ) : (
                  <Button
                    onClick={stopStream}
                    className="bg-red-900 hover:bg-red-800 text-red-400 border border-red-700 font-mono"
                  >
                    <Square className="h-4 w-4 mr-1" /> STOP
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={() => setShowOverlay(!showOverlay)}
                  disabled={!isStreaming}
                  className="border-green-800 text-green-400 hover:bg-green-950 font-mono"
                >
                  <Eye className="h-4 w-4 mr-1" /> OVERLAY: {showOverlay ? "ON" : "OFF"}
                </Button>
              </div>

              <div className="text-xs text-green-600">
                <Zap className="h-3 w-3 inline mr-1" />
                YOLOv8-NANO | O(n) | MPS/GPU
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Detection Log */}
        <div className="lg:col-span-1 space-y-4">
          {/* System Status */}
          <div className="border border-green-800 p-2">
            <div className="text-xs text-green-600 border-b border-green-900 pb-1 mb-2">
              &gt; SYSTEM_STATUS
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">OBJECTS:</span>
                <span>{result?.object_count || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-600">RISK_LVL:</span>
                <span className={result ? riskColors[result.navigation_recommendation.risk_level] : "text-green-400"}>
                  {(result?.navigation_recommendation.risk_level || "safe").toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-600">ALERTS:</span>
                <span className={result?.safety_alerts.length ? "text-red-400" : ""}>
                  {result?.safety_alerts.length || 0}
                </span>
              </div>
            </div>
          </div>

          {/* Safe Directions */}
          <div className="border border-cyan-900 p-2">
            <div className="text-xs text-cyan-500 border-b border-cyan-900 pb-1 mb-2">
              &gt; SAFE_DIRECTIONS
            </div>
            <div className="grid grid-cols-3 gap-1 text-xs">
              {(["left", "center", "right"] as const).map((dir) => {
                const isSafe = result?.navigation_recommendation.safe_directions[dir] ?? true;
                return (
                  <div
                    key={dir}
                    className={`border px-1.5 py-1 text-center ${isSafe
                        ? "border-green-600 text-green-400 bg-green-950/20"
                        : "border-red-700 text-red-400 bg-red-950/20"
                      }`}
                  >
                    <div className="text-[10px] text-green-700">{dir.toUpperCase()}</div>
                    <div>{isSafe ? "CLEAR" : "BLOCKED"}</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-2 text-xs text-green-600">
              RECOMMENDED:{" "}
              <span className="text-green-300">
                {(result?.navigation_recommendation.direction || "none").toUpperCase()}
              </span>
            </div>
          </div>

          {/* Detection Log */}
          <div className="border border-green-800 p-2">
            <div className="text-xs text-green-600 border-b border-green-900 pb-1 mb-2">
              &gt; DETECTION_LOG
            </div>
            <div className="h-48 overflow-y-auto text-xs space-y-0.5">
              {result?.detections.length ? (
                result.detections
                  .sort((a, b) => b.priority - a.priority)
                  .map((d, i) => (
                    <div key={i} className="flex items-center gap-2 py-0.5 border-b border-green-950">
                      <span className="text-green-600">[{i + 1}]</span>
                      <span className={priorityColors[d.priority] || "text-green-400"}>
                        {d.class_name.toUpperCase()}
                      </span>
                      <span className="text-green-600 ml-auto">
                        {Math.round(d.confidence * 100)}%
                      </span>
                    </div>
                  ))
              ) : (
                <div className="text-green-700 text-center py-4">
                  NO_OBJECTS_DETECTED
                </div>
              )}
            </div>
          </div>

          {/* Safety Alerts */}
          <div className="border border-red-900 p-2">
            <div className="text-xs text-red-600 border-b border-red-900 pb-1 mb-2 flex items-center gap-2">
              <AlertTriangle className="h-3 w-3" />
              &gt; SAFETY_ALERTS
            </div>
            <div className="h-32 overflow-y-auto text-xs space-y-1">
              {result?.safety_alerts.length ? (
                result.safety_alerts.map((alert, i) => (
                  <div key={i} className="border-l-2 border-red-700 pl-2 py-1">
                    <div className="flex items-center gap-2">
                      <span className="text-red-400 font-bold">
                        {alert.object.toUpperCase()}
                      </span>
                      <span className={`text-xs ${alert.urgency === "critical" ? "text-red-500 animate-pulse" :
                        alert.urgency === "high" ? "text-orange-500" : "text-yellow-500"
                        }`}>
                        [{alert.urgency.toUpperCase()}]
                      </span>
                    </div>
                    <div className="text-green-600">
                      {alert.position} | {alert.distance}
                    </div>
                    {alert.action_required && (
                      <div className="text-red-500 mt-1">! ACTION_REQUIRED</div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-green-700 text-center py-4">
                  NO_ALERTS
                </div>
              )}
            </div>
          </div>

          {/* Priority Legend */}
          <div className="border border-green-800 p-2">
            <div className="text-xs text-green-600 border-b border-green-900 pb-1 mb-2">
              &gt; PRIORITY_SCALE
            </div>
            <div className="text-xs space-y-0.5">
              <div className="flex justify-between"><span className="text-red-400">P10</span><span>HUMAN</span></div>
              <div className="flex justify-between"><span className="text-orange-400">P9</span><span>ANIMAL</span></div>
              <div className="flex justify-between"><span className="text-yellow-400">P8</span><span>VEHICLE</span></div>
              <div className="flex justify-between"><span className="text-cyan-400">P7</span><span>TRAFFIC</span></div>
              <div className="flex justify-between"><span className="text-gray-400">P3-5</span><span>OBSTACLE</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-4 border border-green-800 p-2 text-xs text-green-600">
        <div className="flex items-center justify-between">
          <span>AUTONOMOUS_NAVIGATION_SYSTEM | REALTIME_SEGMENTATION</span>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Radio className="h-3 w-3" /> WEBCAM_INPUT
            </span>
            <span className="flex items-center gap-1">
              <Target className="h-3 w-3" /> COCO_80_CLASSES
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
