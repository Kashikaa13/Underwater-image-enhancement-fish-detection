
import { useEffect, useState } from "react";

// Component to create animated bubbles in the background
const UnderwaterBackground = () => {
  const [bubbles, setBubbles] = useState<Array<{ id: number; size: number; left: string; delay: number; duration: number }>>([]);
  
  useEffect(() => {
    // Create random bubbles
    const newBubbles = Array.from({ length: 15 }, (_, i) => ({
      id: i,
      size: Math.floor(Math.random() * 20) + 5, // 5-25px bubbles
      left: `${Math.floor(Math.random() * 100)}%`,
      delay: Math.random() * 10, // Random delay up to 10s
      duration: Math.random() * 8 + 4, // 4-12s duration
    }));
    
    setBubbles(newBubbles);
  }, []);
  
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {bubbles.map((bubble) => (
        <div
          key={bubble.id}
          className="bubble absolute bottom-0"
          style={{
            width: `${bubble.size}px`,
            height: `${bubble.size}px`,
            left: bubble.left,
            animationDelay: `${bubble.delay}s`,
            animationDuration: `${bubble.duration}s`,
          }}
        />
      ))}
    </div>
  );
};

export default UnderwaterBackground;
