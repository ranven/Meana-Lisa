"use client";
import { Button } from "@/components/ui/button";
// @ts-ignore
import GaugeChart from 'react-gauge-chart';
import { useState } from 'react';
import { RefreshCwIcon } from "lucide-react";
import { MousePointer2 } from "lucide-react";

interface SimilarPainting {
  _id: string;
  objectID: number;
  isHighlight: boolean;
  primaryImage: string;
  department: string;
  objectName: string;
  title: string;
  artistDisplayName: string;
  artistNationality: string;
  artistBeginDate: number;
  artistEndDate: number;
  artistWikidata_URL: string;
  objectBeginDate: number;
  objectEndDate: number;
  medium: string;
  dimensions: string;
  classification: string;
  objectURL: string;
  [key: string]: any; // For palette[0][0], palette[0][1], etc.
}

interface SimilarPalette {
  similar: SimilarPainting[];
  different: SimilarPainting[];
}

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
  similar_palette: SimilarPalette;
}

interface ResultsViewProps {
  data: AnalysisResult[][] | null;
  imageUrl: string;
  onReset: () => void;
}

export function ResultsView({ data, imageUrl, onReset }: ResultsViewProps) {
  const [selectedPainting, setSelectedPainting] = useState<SimilarPainting | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

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

  const openModal = (painting: SimilarPainting) => {
    setSelectedPainting(painting);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedPainting(null);
  };

  // Function to extract palette data from the API format
  const extractPalette = (painting: SimilarPainting): [string, number][] => {
    const palette: [string, number][] = [];
    let index = 0;
    
    while (painting[`palette[${index}][0]`] && painting[`palette[${index}][1]`]) {
      palette.push([
        painting[`palette[${index}][0]`],
        painting[`palette[${index}][1]`]
      ]);
      index++;
    }
    
    return palette;
  };
  
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

        <div className="flex flex-col items-end mt-8">
          <h3 className="text-lg font-medium text-gray-800">Overall accuracy score</h3>
          <div className="flex flex-row mt-2">
            <div className="flex flex-col items-end gap-3">
              
              <div className="w-48 ">
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
            <div className="flex flex-col items-end gap-3 mt-2">
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
        </div>
      </div>

      {/* 2nd section data */}
      <div className="w-full">
        <h3 className="text-lg font-medium text-gray-800 mb-6">Analogous paintings</h3>
        <div className="grid grid-cols-5 gap-4">
          {result.similar_palette?.similar?.slice(0, 10).map((painting: SimilarPainting, index: number) => (
            <div key={painting._id} className="flex flex-col items-center gap-2">
              <div className="w-20 h-20 rounded-lg overflow-hidden shadow-md hover:shadow-lg hover:scale-110 transition-all duration-200 cursor-pointer">
                <img
                  src={painting.primaryImage}
                  alt={painting.title}
                  className="w-full h-full object-cover"
                  onClick={() => openModal(painting)}
                />
              </div>
              <div className="text-center">
                <p className="text-xs font-medium text-gray-800 truncate max-w-20" title={painting.title}>
                  {painting.title}
                </p>
                <p className="text-xs text-gray-500 truncate max-w-20" title={painting.artistDisplayName}>
                  {painting.artistDisplayName}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Complementary paintings section */}
      <div className="w-full">
        <h3 className="text-lg font-medium text-gray-800 mb-6">Complementary paintings</h3>
        <div className="grid grid-cols-5 gap-4">
          {result.similar_palette?.different?.slice(0, 10).map((painting: SimilarPainting, index: number) => (
            <div key={painting._id} className="flex flex-col items-center gap-2">
              <div className="w-20 h-20 rounded-lg overflow-hidden shadow-md hover:shadow-lg hover:scale-110 transition-all duration-200 cursor-pointer">
                <img
                  src={painting.primaryImage}
                  alt={painting.title}
                  className="w-full h-full object-cover"
                  onClick={() => openModal(painting)}
                />
              </div>
              <div className="text-center">
                <p className="text-xs font-medium text-gray-800 truncate max-w-20" title={painting.title}>
                  {painting.title}
                </p>
                <p className="text-xs text-gray-500 truncate max-w-20" title={painting.artistDisplayName}>
                  {painting.artistDisplayName}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reset Button */}
      <Button 
        onClick={onReset}
        className="mt-8 bg-[#7546b9] hover:bg-[#5a3489] text-white px-6 py-2"
      >
        <RefreshCwIcon className="w-4 h-4" />
        Analyze Another Painting
      </Button>

      {/* Modal */}
      {isModalOpen && selectedPainting && (
        <div className="fixed inset-0  bg-opacity-30 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              {/* Modal Header */}
              <div className="flex justify-between items-start mb-4">
                <h2 className="text-2xl font-bold text-gray-800">{selectedPainting.title}</h2>
                <button
                  onClick={closeModal}
                  className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
                >
                  ×
                </button>
              </div>

              {/* Modal Content */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Image */}
                <div className="flex-shrink-0">
                  <img
                    src={selectedPainting.primaryImage}
                    alt={selectedPainting.title}
                    className="w-full h-auto rounded-lg shadow-lg"
                  />
                  
                  {/* Color palette */}
                  {(() => {
                    const palette = extractPalette(selectedPainting);
                    console.log('Extracted palette:', palette);
                    console.log('Selected painting keys:', Object.keys(selectedPainting).filter(k => k.startsWith('palette')));
                    return (
                      <div className="mt-4">
                        <h3 className="text-sm font-medium text-gray-700 mb-2">Color Palette</h3>
                        {palette.length > 0 ? (
                          <>
                            <div className="flex gap-2 mb-2">
                              {palette.slice(0, 5).map((paletteColor: [string, number], index: number) => (
                                <div
                                  key={index}
                                  className="w-8 h-8 rounded-full border-2 border-gray-300"
                                  style={{ backgroundColor: paletteColor[0] }}
                                  title={`${paletteColor[0]} (${paletteColor[1].toFixed(1)}%)`}
                                />
                              ))}
                            </div>
                            <p className="text-xs text-gray-500">
                              {`[${palette.slice(0, 5).map((color: [string, number]) => color[0]).join(', ')}]`}
                            </p>
                          </>
                        ) : (
                          <p className="text-xs text-gray-400">No palette data available</p>
                        )}
                        
                      </div>
                    );
                  })()}
                </div>

                {/* Details */}
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold text-gray-700 mb-1">Artist</h3>
                    <p className="text-gray-600">{selectedPainting.artistDisplayName}</p>
                    <p className="text-sm text-gray-500">{selectedPainting.artistNationality}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-1">Date</h3>
                    <p className="text-gray-600">
                      {selectedPainting.objectBeginDate === selectedPainting.objectEndDate
                        ? selectedPainting.objectBeginDate
                        : `${selectedPainting.objectBeginDate} - ${selectedPainting.objectEndDate}`}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-1">Medium</h3>
                    <p className="text-gray-600">{selectedPainting.medium}</p>
                  </div>

                  

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-1">Department</h3>
                    <p className="text-gray-600">{selectedPainting.department}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-1">Classification</h3>
                    <p className="text-gray-600">{selectedPainting.classification}</p>
                  </div>

                  {selectedPainting.isHighlight && (
                    <div className="bg-yellow-100 border border-yellow-300 rounded-lg p-3">
                      <p className="text-yellow-800 font-medium">⭐ Museum Highlight</p>
                    </div>
                  )}
                  
                  

                  {/* External Link */}
                  <div className="pt-4">
                    <Button
                      onClick={() => window.open(selectedPainting.objectURL, '_blank')}
                      className="w-full bg-[#7546b9] hover:bg-[#5a3489] text-white"
                    >
                      <MousePointer2 className="w-4 h-4" />
                      View on Met Museum Website
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
