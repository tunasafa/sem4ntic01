import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Semantic Segmentation for Autonomous Navigation",
  description: "Real-time semantic segmentation system for autonomous robot navigation using YOLOv8-Seg. Detects and classifies objects to enable safe navigation while avoiding obstacles, people, animals, and vehicles.",
  keywords: [
    "Semantic Segmentation",
    "Autonomous Navigation",
    "Robot Vision",
    "YOLOv8",
    "Computer Vision",
    "Real-time Processing",
    "Next.js",
    "TypeScript",
    "FastAPI",
    "Machine Learning",
  ],
  authors: [{ name: "Safa" }],
  openGraph: {
    title: "Semantic Segmentation for Autonomous Navigation",
    description: "Real-time object detection and classification for autonomous robot navigation",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Semantic Segmentation for Autonomous Navigation",
    description: "Real-time object detection and classification for autonomous robot navigation",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
