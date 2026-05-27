export default function HeroSection() {
    return (
        <div className="text-center md:text-left space-y-8">

            {/* Main Heading */}
            <div className="space-y-2">
                <h1 className="text-5xl md:text-7xl font-black text-foreground tracking-tight leading-[1.1]">
                    AI Image <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-purple-500 to-blue-600 animate-gradient-x">
                        Analysis Tool
                    </span>
                </h1>
            </div>

            {/* Description */}
            <p className="text-lg md:text-xl text-muted-foreground max-w-lg mx-auto md:mx-0 leading-relaxed font-medium">
                Experience the power of <span className="text-foreground font-semibold">Dual-AI Architecture</span>.
                Instantly segment objects and classify images with professional precision.
            </p>

            {/* Stats/Trust Indicators */}
            <div className="pt-4 flex items-center justify-center md:justify-start gap-8 text-sm font-semibold text-muted-foreground">
                <div className="flex items-center gap-2">
                    <div className="w-1 h-4 bg-primary rounded-full" />
                    UNet Segmentation
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-1 h-4 bg-purple-500 rounded-full" />
                    ResNet50 Classify
                </div>
            </div>
        </div>
    );
}
