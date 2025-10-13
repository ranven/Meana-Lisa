"use client";
import { LoaderPinwheel } from "lucide-react";

export function LoaderView() {
  return (
    <div className="flex flex-col items-center justify-center gap-8 min-h-screen">
      {/* Loading Spinner */}
      <div className="relative">
        <LoaderPinwheel className="w-16 h-16 text-black animate-spin" />
      </div>
      
      {/* Loading Text */}
      <p className="text-[22px] font-light text-black">Analyzing painting</p>
    </div>
  );
}
