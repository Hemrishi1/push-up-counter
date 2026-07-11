import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';

import api from '../services/api';
import { Play, Square, Activity, Target } from 'lucide-react';

const LiveWorkout = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stats, setStats] = useState({ count: 0, fps: 0, percent: 0 });
  const [duration, setDuration] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Audio for feedback
  const beepSound = useRef(new Audio('data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU...')); // Simplified placeholder or user provides file

  useEffect(() => {
    let interval: number;
    if (isStreaming) {
      interval = window.setInterval(() => setDuration(d => d + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isStreaming]);

  const startWorkout = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(e => console.error("Video play error:", e));
        }
      streamRef.current = stream;
      setIsStreaming(true);
      
      const token = localStorage.getItem('access_token');
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
      const defaultWsUrl = apiUrl.replace(/^http/, 'ws') + '/workout/stream';
      const wsUrl = import.meta.env.VITE_WS_URL || defaultWsUrl;
      console.log('Connecting to WebSocket:', `${wsUrl}/${token}`);
      wsRef.current = new WebSocket(`${wsUrl}/${token}`);
      
      let isFramePending = false;
      
      wsRef.current.onmessage = (event) => {
        isFramePending = false;
        const data = JSON.parse(event.data);
        if (data.type === 'result') {
          setStats({ count: data.count, fps: data.fps, percent: data.percent });
          
          if (data.event === 'rep_complete') {
            try { beepSound.current.play(); } catch(e) {}
          }
          
          // Draw image on canvas
          const img = new Image();
          img.onload = () => {
            const ctx = canvasRef.current?.getContext('2d');
            if (ctx && canvasRef.current) {
              ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
            }
          };
          img.src = data.data;
        }
      };

      // Start sending frames
      const sendFrames = () => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        
        if (!isFramePending) {
          const canvas = document.createElement('canvas');
          canvas.width = 640;
          canvas.height = 480;
          const ctx = canvas.getContext('2d');
          if (ctx && videoRef.current && videoRef.current.readyState >= 2) { // 2 = HAVE_CURRENT_DATA
            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            const base64Data = canvas.toDataURL('image/jpeg', 0.5);
            wsRef.current.send(JSON.stringify({ type: 'frame', data: base64Data }));
            isFramePending = true;
          }
        }
        
        requestAnimationFrame(sendFrames);
      };
      
      wsRef.current.onopen = () => {
        sendFrames();
      };
      
    } catch (err: any) {
      console.error("Error accessing camera:", err);
      alert("Could not access the camera. Please ensure you have granted camera permissions to this site and that a camera is connected. (" + err.message + ")");
    }
  };

  const stopWorkout = async () => {
    setIsStreaming(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    // Save workout
    if (stats.count > 0) {
      try {
        await api.post('/workout/end', {
          pushups: stats.count,
          duration: duration,
          calories: stats.count * 0.3, // Simple estimation
          accuracy: 95.0, // Placeholder
          average_speed: (stats.count / duration) * 60 || 0
        });
        setStats({ count: 0, fps: 0, percent: 0 });
        setDuration(0);
        alert('Workout saved!');
      } catch(e) {
        console.error(e);
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Workout</h1>
          <p className="text-muted-foreground">Ensure your full body is visible in the camera.</p>
        </div>
        {!isStreaming ? (
          <Button onClick={startWorkout} size="lg" className="bg-green-600 hover:bg-green-700">
            <Play className="mr-2 h-5 w-5" /> Start Workout
          </Button>
        ) : (
          <Button onClick={stopWorkout} size="lg" variant="destructive">
            <Square className="mr-2 h-5 w-5" /> End Workout
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="overflow-hidden border-2 border-primary/20 bg-black">
            <div className="relative aspect-video flex items-center justify-center bg-zinc-900">
              <video 
                ref={videoRef} 
                className="absolute inset-0 w-full h-full object-cover opacity-0" 
                autoPlay 
                playsInline 
                muted
              />
              <canvas 
                ref={canvasRef} 
                width={1280} 
                height={720} 
                className="w-full h-full object-cover"
              />
              {!isStreaming && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500">
                  <Activity size={48} className="mb-4 opacity-50" />
                  <p>Camera is off. Click start to begin.</p>
                </div>
              )}
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Real-time Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-end border-b border-border/50 pb-4">
                <span className="text-muted-foreground">Pushups</span>
                <span className="text-5xl font-bold text-primary">{Math.floor(stats.count)}</span>
              </div>
              <div className="flex justify-between items-end border-b border-border/50 pb-4">
                <span className="text-muted-foreground">Time</span>
                <span className="text-3xl font-bold">{formatTime(duration)}</span>
              </div>
              <div className="flex justify-between items-end pb-2">
                <span className="text-muted-foreground">Calories</span>
                <span className="text-3xl font-bold text-orange-500">{(stats.count * 0.3).toFixed(1)}</span>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Form Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-4">
                <div className="relative flex items-center justify-center w-32 h-32">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle className="text-muted stroke-current" strokeWidth="8" cx="50" cy="50" r="40" fill="transparent" />
                    <circle 
                      className="text-primary stroke-current transition-all duration-300" 
                      strokeWidth="8" 
                      strokeLinecap="round"
                      cx="50" cy="50" r="40" fill="transparent" 
                      strokeDasharray="251.2" 
                      strokeDashoffset={251.2 - (251.2 * stats.percent) / 100}
                    />
                  </svg>
                  <div className="absolute text-2xl font-bold">{stats.percent}%</div>
                </div>
                <div className="mt-4 text-sm text-muted-foreground flex items-center">
                  <Target size={16} className="mr-2" />
                  Target: 90% and above
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default LiveWorkout;
