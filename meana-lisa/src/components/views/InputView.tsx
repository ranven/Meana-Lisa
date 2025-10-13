"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";

interface InputViewProps {
  onAnalyze: (imageUrl: string) => void;
  onUpload: (file: File) => void;
}

export function InputView({ onAnalyze, onUpload }: InputViewProps) {
  const [imageUrl, setImageUrl] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (imageUrl.trim()) {
      onAnalyze(imageUrl.trim());
    }
  };

  const handleFileUpload = (file: File) => {
    onUpload(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center gap-8 py-20 mt-6">
      {/* Upload Section */}
      <div className="flex flex-col items-center gap-4">
        <p className="text-[22px] font-light text-black">Begin by uploading a painting</p>
        
        <div
          className={`w-96 h-[238px] border-1 bg-[#D9D9D9] border-dashed rounded-lg flex flex-col items-center justify-center gap-4 cursor-pointer transition-colors ${
            dragActive 
              ? "border-[#7546b9] bg-purple-50" 
              : "border-black hover:border-gray-400"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <Upload
            className="w-12 h-12 text-[#646E7B]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
          </Upload>
          <p className="text-[#646E7B]">Click to upload</p>
        </div>
        
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />
        
        <Button 
          onClick={() => document.getElementById('file-upload')?.click()}
          className="w-full cursor-pointer bg-[#0F172A] hover:bg-[#7546b9] text-white px-6 py-2"
        >
          Continue
        </Button>
      </div>

      {/* URL Section */}
      <div className="flex flex-col items-center gap-4 mt-10">
        <p className="text-[22px] font-light text-black">Or paste an Image URL</p>
        
        <form onSubmit={handleUrlSubmit} className="flex gap-2 w-[90%]">
          <Input
            type="url"
            placeholder="Paste Image URL"
            value={imageUrl}
            onChange={(e) => setImageUrl(e.target.value)}
            className="w-80"
          />
          <Button 
            type="submit"
            className="bg-[#0F172A] cursor-pointer hover:bg-[#5a3489] text-white px-6 py-2"
          >
            Continue
          </Button>
        </form>
      </div>
    </div>
  );
}
