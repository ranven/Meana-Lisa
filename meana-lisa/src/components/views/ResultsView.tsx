"use client";
import { Button } from "@/components/ui/button";
// @ts-ignore
import GaugeChart from 'react-gauge-chart';

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

interface ResultsViewProps {
  data: AnalysisResult[][] | null;
  imageUrl: string;
  onReset: () => void;
}

export function ResultsView({ data, imageUrl, onReset }: ResultsViewProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-20">
        <p className="text-lg text-gray-600">No analysis data available</p>
        <Button onClick={onReset} className="bg-[#7546b9] hover:bg-[#5a3489] text-white">
          Try Again
        </Button>
      </div>
    );
  }

  // Extract the first result from the nested array structure
  const result = data[0][0];
  
  // Calculate average accuracy
  const averageAccuracy = (result.department.accuracy + result.century.accuracy + result.nat.accuracy) / 3;
  const accuracyPercentage = Math.round(averageAccuracy * 100);

  return (
    <div className="flex flex-col max-w-4xl justify-self-center items-center gap-8 py-10">


      {/* Main Content Area */}
      <div className="flex gap-8 ">
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
            from the <span className="font-bold text-black">{result.department.value}</span> department
          </p>
          
          <p className="text-gray-500">
            It was likely painted in...
          </p>
          <p>
            the <span className="font-bold text-black">{result.century.value}th century</span>
          </p>
          
          <p className="text-gray-500">
            By someone who was
          </p>
          <p>
            <span className="font-bold text-black">{result.nat.value}</span>
          </p>
        </div>
      </div>


      {/* 1st section data */}
      <div className="flex flex-row justify-between w-full">

        {/* Colour palette */}
        <div className="flex flex-col items-start gap-3 mt-8">
          <h3 className="text-lg font-medium text-gray-800">Primary colour palette</h3>
          <div className="flex gap-2">
            {result.palette.slice(0, 5).map((paletteColor: [string, number], index: number) => (
              <div
                key={index}
                className="w-10 h-10 rounded-full border-2 border-gray-300"
                style={{ backgroundColor: paletteColor[0] }}
                title={`${paletteColor[0]} (${paletteColor[1].toFixed(1)}%)`}
              />
            ))}
          </div>
          <p className="text-[13px] text-gray-500">
            {`[${result.palette.slice(0, 5).map((color: [string, number]) => color[0]).join(', ')}]`}
          </p>
        </div>

        {/* Accuracy Score Gauge Chart */}
        <div className="flex flex-col items-end gap-3 mt-8">
          <h3 className="text-lg font-medium text-gray-800">Overall Accuracy Score</h3>
          <div className="w-48 h-48">
            <GaugeChart
              id="accuracy-gauge"
              nrOfLevels={15}
              percent={averageAccuracy}
              colors={["#f9b3ff", "#f36eff", "#ef45ff", "#ea23fc"]}
              arcWidth={0.3}
              textColor="#000"
              needleColor="#374151"
              animate={true}
              animateDuration={1000}
            />
          </div>
        </div>

        {/* Accuracy scores */}
        <div className="flex flex-col items-end gap-3 mt-8">
          <div className="gap-2 text-end text-[15px]">
            <p className=" text-gray-500">
              Department: <span className="font-bold text-black">{(result.department.accuracy * 100).toFixed(0)}%</span>
            </p>
            <p className=" text-gray-500">
              Century: <span className="font-bold text-black">{(result.century.accuracy * 100).toFixed(0)}%</span>
            </p>
            <p className=" text-gray-500">
              Nationality: <span className="font-bold text-black">{(result.nat.accuracy * 100).toFixed(0)}%</span>
            </p>
          </div>
        </div>

        
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
