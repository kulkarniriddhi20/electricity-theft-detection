import React from "react";
import { useNavigate } from "react-router-dom";
import bgImage from "../assets/bg.jpg";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div
      className="flex flex-col items-center justify-center min-h-screen bg-cover bg-center relative"
      style={{ backgroundImage: `url(${bgImage})` }}
    >
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/40"></div>

      {/* Highlighted Text Card */}
      <div className="relative z-10 bg-white/90 backdrop-blur-md p-12 rounded-xl shadow-lg text-center max-w-2xl animate-fadeIn">
        <h1 className="text-5xl font-bold mb-6 text-blue-700 animate-grow">
          Electricity Theft Detection
        </h1>
        <p className="text-lg text-gray-700 mb-8 animate-shimmer">
          Detect electricity theft in real-time using advanced analytics and
          machine learning. Monitor consumption patterns, view reports, and
          act on anomalies immediately.
        </p>
      </div>
    </div>
  );
};

export default Home;