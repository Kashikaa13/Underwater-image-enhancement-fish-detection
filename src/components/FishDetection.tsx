
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useEffect, useRef, useState } from "react";

export interface DetectionResult {
  imageUrl: string;
  detections: Array<{
    id: number;
    species: string;
    confidence: number;
    box: [number, number, number, number]; // [x, y, width, height]
  }>;
}

interface FishDetectionProps {
  result: DetectionResult | null;
  isLoading: boolean;
}

const FishDetection = ({ result, isLoading }: FishDetectionProps) => {
  const [loaded, setLoaded] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    setLoaded(false);
  }, [result]);

  const renderBoundingBoxes = () => {
    if (!result || !loaded || !imageRef.current || !containerRef.current) return null;

    const imgElement = imageRef.current;
    const containerElement = containerRef.current;
    
    const scaleX = imgElement.clientWidth / imgElement.naturalWidth;
    const scaleY = imgElement.clientHeight / imgElement.naturalHeight;
    
    // Calculate offsets to position boxes correctly
    const imgRect = imgElement.getBoundingClientRect();
    const containerRect = containerElement.getBoundingClientRect();
    
    const offsetX = imgRect.left - containerRect.left;
    const offsetY = imgRect.top - containerRect.top;
    
    return result.detections.map((detection) => {
      const [x, y, width, height] = detection.box;
      
      // Scale and position the bounding box
      const boxStyle = {
        left: offsetX + x * scaleX,
        top: offsetY + y * scaleY,
        width: width * scaleX,
        height: height * scaleY,
      };
      
      // Position the label at the top of the box
      const labelStyle = {
        left: boxStyle.left,
        top: boxStyle.top - 25, // Position above the box
      };
      
      return (
        <div key={detection.id}>
          <div className="fish-detect-box" style={boxStyle}></div>
          <div className="fish-detect-label" style={labelStyle}>
            {detection.species} {(detection.confidence * 100).toFixed(1)}%
          </div>
        </div>
      );
    });
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 pt-4">
        <CardTitle className="text-center">Fish Detection Results</CardTitle>
      </CardHeader>
      <CardContent className="flex-grow p-4">
        <div 
          ref={containerRef}
          className="w-full h-full relative flex justify-center items-center"
        >
          {isLoading ? (
            <Skeleton className="w-full aspect-video" />
          ) : result ? (
            <>
              <img
                ref={imageRef}
                src={result.imageUrl}
                alt="Fish detection results"
                className="max-h-full max-w-full object-contain rounded"
                onLoad={() => setLoaded(true)}
              />
              {renderBoundingBoxes()}
            </>
          ) : (
            <div className="w-full aspect-video bg-muted flex items-center justify-center rounded-md">
              <p className="text-muted-foreground text-sm">Detection results will appear here</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default FishDetection;
