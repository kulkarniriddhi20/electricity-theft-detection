import React from "react";
import bgImage from "../assets/bg.jpg";

const About = () => {
  return (
    <div
      className="flex flex-col items-center justify-center min-h-[80vh] bg-cover bg-center relative"
      style={{ backgroundImage: `url(${bgImage})` }}
    >
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/40"></div>

      {/* Highlighted Card */}
      <div className="relative z-10 bg-white/90 backdrop-blur-md p-12 rounded-xl shadow-lg text-center max-w-2xl animate-fadeIn">
        <h1 className="text-4xl font-bold mb-4 text-blue-700 animate-grow">
          About Us
        </h1>
        <p className="text-lg text-black animate-shimmer">
          Our Electricity Theft Detection platform helps utilities identify
          unusual consumption patterns in real-time using advanced analytics
          and machine learning. Monitor, report, and act on anomalies quickly.
        </p>
      </div>
    </div>
  );
};

export default About;