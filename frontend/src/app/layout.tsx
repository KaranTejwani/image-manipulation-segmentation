"use client";

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Navbar, NavBody, NavItems, MobileNav, MobileNavHeader, MobileNavToggle, NavbarSettings } from "@/components/ui/resizable-navbar";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import ThemeToggle from "@/components/ThemeToggle";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

function NavbarWrapper() {
  const [isOpen, setIsOpen] = useState(false);


  return (
    <Navbar>
      <NavBody>
        {/* Logo */}
        <a href="#" className="relative z-20 flex items-center gap-2 px-2 py-1">
          <div className="p-1.5 bg-primary rounded-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-foreground">AI Image Analysis</span>
        </a>



        {/* Right Actions */}
        <div className="relative z-20 flex items-center gap-2">
          <NavbarSettings />
          <ThemeToggle />
        </div>
      </NavBody>

      {/* Mobile Nav */}
      <MobileNav>
        <MobileNavHeader>
          <a href="#" className="flex items-center gap-2">
            <div className="p-1.5 bg-primary rounded-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-foreground">AI Image Analysis</span>
          </a>
          <div className="flex items-center gap-2">
            <NavbarSettings />
            <ThemeToggle />
            <MobileNavToggle isOpen={isOpen} onClick={() => setIsOpen(!isOpen)} />
          </div>
        </MobileNavHeader>
      </MobileNav>
    </Navbar>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <NavbarWrapper />
        {children}
      </body>
    </html>
  );
}
