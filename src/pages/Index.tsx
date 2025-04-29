import { useState } from "react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import ImageUploader from "@/components/ImageUploader";
import EnhancedImage from "@/components/EnhancedImage";
import FishDetection, { DetectionResult } from "@/components/FishDetection";
import UnderwaterBackground from "@/components/UnderwaterBackground";
import Header from "@/components/Header";
import { Loader } from "lucide-react";

// Mock data for demonstration - this would come from your API in a real app
const mockEnhancedImages = {
  usrgan: "https://images.unsplash.com/photo-1518877593221-1f28583780b4",
};

// Mock detection data
const mockDetectionResult: DetectionResult = {
  imageUrl: "https://images.unsplash.com/photo-1518877593221-1f28583780b4",
  detections: [
    { 
      id: 1, 
      species: "Clownfish", 
      confidence: 0.97, 
      box: [50, 50, 200, 100] 
    },
    { 
      id: 2, 
      species: "Blue Tang", 
      confidence: 0.86, 
      box: [150, 150, 220, 120] 
    }
  ]
};

const Index = () => {
  const [originalImage, setOriginalImage] = useState<File | null>(null);
  const [enhancedImages, setEnhancedImages] = useState<{
    usrgan: string | null;
  }>({
    usrgan: null,
  });
  
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("enhancement");

  const handleImageUpload = (file: File) => {
    setOriginalImage(file);
    // Reset results when new image is uploaded
    setEnhancedImages({
      usrgan: null,
    });
    setDetectionResult(null);
  };

  const handleProcessImage = async () => {
    if (!originalImage) {
      toast.error("Please upload an image first");
      return;
    }

    setIsProcessing(true);

    try {
      // Simulate API call for image enhancement
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Update with mock data for demo purposes
      setEnhancedImages({
        usrgan: mockEnhancedImages.usrgan,
      });
      
      // Simulate API call for detection
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setDetectionResult({
        ...mockDetectionResult,
        imageUrl: URL.createObjectURL(originalImage), // In real app, this would be API response
      });
      
      toast.success("Processing completed successfully");
      
    } catch (error) {
      toast.error("Error processing the image");
      console.error("Processing error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen pb-16 underwater-pattern">
      <UnderwaterBackground />
      
      <div className="container max-w-6xl">
        <Header />
        
        <div className="mt-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <div className="sticky top-6 space-y-6">
                <Card className="p-6 shadow-lg bg-white/70 backdrop-blur-sm">
                  <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
                  <ImageUploader 
                    onImageUpload={handleImageUpload}
                    isProcessing={isProcessing}
                  />
                  
                  <div className="mt-6">
                    <Button 
                      className="w-full"
                      onClick={handleProcessImage}
                      disabled={!originalImage || isProcessing}
                    >
                      {isProcessing ? (
                        <>
                          <Loader className="mr-2 h-4 w-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        "Process Image"
                      )}
                    </Button>
                  </div>
                </Card>
              </div>
            </div>
            
            <div className="lg:col-span-2">
              <Card className="p-6 shadow-lg bg-white/70 backdrop-blur-sm">
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="enhancement">Image Enhancement</TabsTrigger>
                    <TabsTrigger value="detection">Fish Detection</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="enhancement" className="mt-6">
                    <h2 className="text-xl font-semibold mb-4">Enhanced Image</h2>
                    <div className="grid grid-cols-1 gap-4">
                      <EnhancedImage
                        title="USRGAN Enhancement"
                        imageUrl={enhancedImages.usrgan}
                        isLoading={isProcessing}
                      />
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="detection" className="mt-6">
                    <FishDetection 
                      result={detectionResult} 
                      isLoading={isProcessing} 
                    />
                  </TabsContent>
                </Tabs>
              </Card>
            </div>
          </div>
        </div>
        
        <footer className="mt-16 text-center text-sm text-muted-foreground">
          <p>© 2025 AquaVision - Underwater Image Enhancement & Fish Detection</p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
