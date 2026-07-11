import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Activity, Calendar, Clock, Flame, Target } from 'lucide-react';
import api from '../services/api';

interface Workout {
  id: string;
  pushups: number;
  duration: number;
  calories: number;
  accuracy: number;
  created_at: string;
}

const History = () => {
  const [workouts, setWorkouts] = useState<Workout[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await api.get('/workout/history');
        setWorkouts(response.data);
      } catch (error) {
        console.error('Failed to fetch history:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    });
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Workout History</h1>
          <p className="text-muted-foreground">View your past performances and track your progress.</p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      ) : workouts.length === 0 ? (
        <Card className="glass-card flex flex-col items-center justify-center py-20 text-center space-y-4">
          <div className="p-4 bg-primary/10 rounded-full">
            <Activity size={48} className="text-primary opacity-80" />
          </div>
          <div>
            <h3 className="text-xl font-semibold">No workouts yet</h3>
            <p className="text-muted-foreground mt-2 max-w-sm">
              Your workout history is empty. Go to the Live Workout tab to start your first session!
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {workouts.map((workout) => (
            <Card key={workout.id} className="glass-card hover:bg-white/5 dark:hover:bg-black/20 transition-all duration-300">
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0 border-b border-border/50">
                <CardTitle className="text-lg font-medium flex items-center gap-2">
                  <Calendar size={18} className="text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">{formatDate(workout.created_at)}</span>
                </CardTitle>
                <div className="px-2 py-1 bg-primary/20 text-primary text-xs font-semibold rounded-md">
                  Completed
                </div>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="flex justify-between items-end mb-4">
                  <span className="text-muted-foreground text-sm font-medium">Total Pushups</span>
                  <span className="text-4xl font-bold text-primary">{workout.pushups}</span>
                </div>
                
                <div className="grid grid-cols-3 gap-2 pt-4 border-t border-border/50">
                  <div className="flex flex-col items-center justify-center text-center p-2 rounded-lg bg-black/20">
                    <Clock size={16} className="text-blue-400 mb-1" />
                    <span className="text-xs text-muted-foreground">Time</span>
                    <span className="font-semibold text-sm">{formatDuration(workout.duration)}</span>
                  </div>
                  <div className="flex flex-col items-center justify-center text-center p-2 rounded-lg bg-black/20">
                    <Flame size={16} className="text-orange-500 mb-1" />
                    <span className="text-xs text-muted-foreground">Calories</span>
                    <span className="font-semibold text-sm">{workout.calories.toFixed(1)}</span>
                  </div>
                  <div className="flex flex-col items-center justify-center text-center p-2 rounded-lg bg-black/20">
                    <Target size={16} className="text-green-500 mb-1" />
                    <span className="text-xs text-muted-foreground">Accuracy</span>
                    <span className="font-semibold text-sm">{workout.accuracy.toFixed(0)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
