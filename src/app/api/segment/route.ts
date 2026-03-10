import { NextRequest, NextResponse } from "next/server";
import { logger } from "@/lib/logger";

// Get segmentation service URL from environment variable
const SEGMENTATION_SERVICE_URL = process.env.SEGMENTATION_API_URL || "http://127.0.0.1:3030";

// Proxy to segmentation service
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${SEGMENTATION_SERVICE_URL}/segment`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    logger.error("Segmentation proxy error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to connect to segmentation service" },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const response = await fetch(`${SEGMENTATION_SERVICE_URL}/status`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    logger.error("Status check error:", error);
    return NextResponse.json(
      { status: "error", error: "Service unavailable" },
      { status: 503 }
    );
  }
}
