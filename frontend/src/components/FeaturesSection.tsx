import { Layers, Zap, Scan } from "lucide-react";

export default function FeaturesSection() {
    const features = [
        { icon: Layers, title: "Dual Analysis", desc: "Performs both Segmentation and Classification in one go." },
        { icon: Zap, title: "Advanced Models", desc: "Powered by UNet (EfficientNet) and ResNet50." },
        { icon: Scan, title: "Instant Results", desc: "Get accurate masks and class predictions in seconds." },
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-8">
            {features.map((feature, i) => (
                <div
                    key={i}
                    className="p-6 bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl text-center hover:border-primary/30 hover:bg-card/80 transition-all duration-300 group"
                >
                    <div className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-xl mb-4 group-hover:scale-110 transition-transform duration-300">
                        <feature.icon className="w-6 h-6 text-primary" />
                    </div>
                    <h3 className="font-bold text-foreground mb-2 text-lg">{feature.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{feature.desc}</p>
                </div>
            ))}
        </div>
    );
}
