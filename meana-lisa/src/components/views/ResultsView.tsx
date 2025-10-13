"use client";
import { Button } from "@/components/ui/button";

interface PaletteColor {
  color: string;
  percentage: number;
}

interface AnalysisResult {
  department: string;
  century: number;
  nationality: string;
  palette: PaletteColor[];
}

interface ResultsViewProps {
  data: AnalysisResult | null;
  imageUrl: string;
  onReset: () => void;
}

export function ResultsView({ data, imageUrl, onReset }: ResultsViewProps) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-20">
        <p className="text-lg text-gray-600">No analysis data available</p>
        <Button onClick={onReset} className="bg-[#7546b9] hover:bg-[#5a3489] text-white">
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-8 py-10">


      {/* Main Content Area */}
      <div className="flex gap-8 max-w-4xl">
        {/* Painting Image */}
        <div className="flex-shrink-0">
          <img
            src={imageUrl}
            alt="Analyzed painting"
            className="w-80 h-auto rounded-lg shadow-lg"
          />
        </div>

        {/* Predictions */}
        <div className="flex flex-col justify-center gap-4 text-lg">
          <p className="text-gray-500">
            We predict your painting to be...
          </p>
          <p>
            from the <span className="font-bold text-black">{data.department}</span> department
          </p>
          
          <p className="text-gray-500">
            It was likely painted in...
          </p>
          <p>
            the <span className="font-bold text-black">{data.century}th century</span>
          </p>
          
          <p className="text-gray-500">
            By someone who was
          </p>
          <p>
            <span className="font-bold text-black">{data.nationality}</span>
          </p>
        </div>
      </div>

      {/* Color Palette */}
      <div className="flex flex-col items-center gap-3 mt-8">
        <h3 className="text-lg font-medium text-gray-800">Color palette</h3>
        <div className="flex gap-2">
          {data.palette.slice(0, 5).map((paletteColor, index) => (
            <div
              key={index}
              className="w-8 h-8 rounded-full border-2 border-gray-300"
              style={{ backgroundColor: paletteColor.color }}
              title={`${paletteColor.color} (${paletteColor.percentage.toFixed(1)}%)`}
            />
          ))}
        </div>
        <p className="text-sm text-gray-500">
          Top 5 colors with percentages
        </p>
      </div>

      {/* Reset Button */}
      <Button 
        onClick={onReset}
        className="mt-8 bg-[#7546b9] hover:bg-[#5a3489] text-white px-6 py-2"
      >
        Analyze Another Painting
      </Button>
    </div>
  );
}
