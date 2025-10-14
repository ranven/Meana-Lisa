"use client";
import { useState, useEffect } from "react";
import { InputView } from "@/components/views/InputView";
import { LoaderView } from "@/components/views/LoaderView";
import { ResultsView } from "@/components/views/ResultsView";

type AppState = 'input' | 'loading' | 'results';

interface AnalysisResult {
  department: {
    value: string;
    accuracy: number;
  };
  nat: {
    value: string;
    accuracy: number;
  };
  century: {
    value: number;
    accuracy: number;
  };
  palette: [string, number][];
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>('input');
  const [analysisData, setAnalysisData] = useState<AnalysisResult[] | null>(null);
  const [currentImageUrl, setCurrentImageUrl] = useState<string>("");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [displayState, setDisplayState] = useState<AppState>('input');

  // Handle state transitions with fade effect
  const transitionToState = (newState: AppState) => {
    if (newState === displayState) return;

    setIsTransitioning(true);

    // Start exit transition
    setTimeout(() => {
      setDisplayState(newState);
      setIsTransitioning(false);
    }, 150); // Half of the transition duration
  };

  // Update display state when app state changes
  useEffect(() => {
    transitionToState(appState);
  }, [appState]);

  const handleAnalyze = async (imageUrl: string) => {
    setCurrentImageUrl(imageUrl);
    setAppState('loading');

    try {
      const res = await fetch("/api/url-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: imageUrl }),
      });

      const data = await res.json();

      if (data.data && data.data.length > 0) {
        // Pass the raw API response directly
        setAnalysisData(data.data);
        setAppState('results');
      } else {
        console.error("Analysis failed:", data.error);
        setAppState('input');
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      setAppState('input');
    }
  };

  const handleUpload = async (file: File) => {
    setAppState('loading');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch("/api/upload-predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.data && data.data.length > 0) {
        // Create a preview URL for the uploaded file
        const previewUrl = URL.createObjectURL(file);
        setCurrentImageUrl(previewUrl);
        
        // Pass the raw API response directly
        setAnalysisData(data.data);
        setAppState('results');
      } else {
        console.error("Upload analysis failed:", data.error);
        setAppState('input');
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setAppState('input');
    }
  };

  const handleReset = () => {
    setAppState('input');
    setAnalysisData(null);
    setCurrentImageUrl("");
  };

  const renderCurrentView = () => {
    switch (displayState) {
      case 'input':
        return <InputView onAnalyze={handleAnalyze} onUpload={handleUpload} />;
      case 'loading':
        return <LoaderView />;
      case 'results':
        return (
          <ResultsView
            data={analysisData}
            imageUrl={currentImageUrl}
            onReset={handleReset}
          />
        );
      default:
        return <InputView onAnalyze={handleAnalyze} onUpload={handleUpload} />;
    }
  };

  return (
    <div className="min-h-full">
      <div
        className={`transition-all duration-300 ease-in-out ${isTransitioning ? 'fade-exit-active' : 'fade-enter-active'
          }`}
      >
        {renderCurrentView()}
      </div>
    </div>
  );
}
