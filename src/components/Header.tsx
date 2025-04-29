
import { CircleCheckIcon } from "lucide-react";

const Header = () => {
  return (
    <header className="py-6 text-center">
      <h1 className="text-3xl font-bold text-primary-foreground bg-primary inline-block px-5 py-2 rounded-lg shadow-lg mb-2">
        AquaVision Enhance &amp; Detect
      </h1>
      <p className="text-muted-foreground max-w-2xl mx-auto">
        Underwater image enhancement using advanced CV-GAN techniques and fish species detection
      </p>
      
      <div className="mt-4 flex flex-wrap justify-center gap-4">
        <div className="inline-flex items-center gap-1 bg-secondary px-3 py-1 rounded-full">
          <CircleCheckIcon className="h-4 w-4 text-primary" /> 
          <span className="text-xs">USRGAN Enhancement</span>
        </div>
        <div className="inline-flex items-center gap-1 bg-secondary px-3 py-1 rounded-full">
          <CircleCheckIcon className="h-4 w-4 text-primary" /> 
          <span className="text-xs">Species Detection</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
