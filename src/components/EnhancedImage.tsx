
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface EnhancedImageProps {
  title: string;
  imageUrl: string | null;
  isLoading?: boolean;
}

const EnhancedImage = ({ title, imageUrl, isLoading = false }: EnhancedImageProps) => {
  const [loaded, setLoaded] = useState(false);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 pt-4">
        <CardTitle className="text-center text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-grow flex flex-col justify-center items-center p-4">
        {isLoading ? (
          <div className="w-full aspect-video relative overflow-hidden rounded-md">
            <Skeleton className="w-full h-full absolute inset-0" />
          </div>
        ) : imageUrl ? (
          <div className="relative w-full h-full flex justify-center">
            {!loaded && (
              <div className="absolute inset-0 flex justify-center items-center">
                <Skeleton className="w-full aspect-video" />
              </div>
            )}
            <img
              src={imageUrl}
              alt={title}
              className="max-h-full max-w-full object-contain rounded transition-opacity duration-300"
              style={{ opacity: loaded ? 1 : 0 }}
              onLoad={() => setLoaded(true)}
            />
          </div>
        ) : (
          <div className="w-full aspect-video bg-muted flex items-center justify-center rounded-md">
            <p className="text-muted-foreground text-sm">Result will appear here</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default EnhancedImage;
