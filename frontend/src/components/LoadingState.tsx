import { Activity } from "lucide-react";

export default function LoadingState() {
    return (
        <div className="max-w-4xl mx-auto min-h-[60vh] flex flex-col items-center justify-center">
            <div className="relative mb-8">
                <div className="w-24 h-24 border-4 border-primary/20 rounded-full" />
                <div className="absolute inset-0 w-24 h-24 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                    <Activity className="w-8 h-8 text-primary animate-pulse" />
                </div>
            </div>
            <h2 className="text-3xl font-bold text-foreground mb-3">Analyzing Image</h2>
            <p className="text-muted-foreground text-lg animate-pulse">Running Segmentation & Classification...</p>
        </div>
    );
}
