
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Image } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
  isProcessing: boolean;
}

const ImageUploader = ({ onImageUpload, isProcessing }: ImageUploaderProps) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    const reader = new FileReader();
    
    reader.onload = () => {
      setPreview(reader.result as string);
      onImageUpload(file);
    };
    
    reader.readAsDataURL(file);
  }, [onImageUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    disabled: isProcessing,
    multiple: false
  });

  return (
    <div className="w-full space-y-4">
      <Card className={`border-2 border-dashed rounded-lg p-6 transition-all duration-200 
        ${isDragActive ? 'border-primary bg-primary/10' : 'border-border'} 
        ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-primary/50'}`}>
        <div
          {...getRootProps()}
          className="flex flex-col items-center justify-center space-y-4 text-center h-48"
        >
          <input {...getInputProps()} />
          
          {preview ? (
            <div className="relative w-full h-full flex justify-center">
              <img
                src={preview}
                alt="Preview"
                className="max-h-full max-w-full object-contain rounded"
              />
            </div>
          ) : (
            <>
              <div className="p-3 bg-secondary rounded-full">
                {isDragActive ? (
                  <Image className="h-8 w-8 text-primary animate-pulse" />
                ) : (
                  <Upload className="h-8 w-8 text-primary" />
                )}
              </div>
              <div>
                <p className="text-sm font-medium">
                  {isDragActive 
                    ? "Drop the image here" 
                    : "Drag & drop an underwater image here"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse files (JPG, PNG)
                </p>
              </div>
            </>
          )}
        </div>
      </Card>
      {preview && (
        <div className="flex justify-center">
          <Button 
            variant="outline" 
            onClick={() => setPreview(null)}
            disabled={isProcessing}
          >
            Remove image
          </Button>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
