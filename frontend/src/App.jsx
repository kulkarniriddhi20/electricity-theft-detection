import { BrowserRouter, Routes, Route } from "react-router-dom";
import React, { useState } from "react";

import Navbar from "./pages/Navbar";
import Footer from "./pages/Footer";

import Home from "./pages/Home";
import Login from "./pages/Login";
import Upload from "./pages/Upload";
import Analytics from "./pages/Analytics";
import About from "./pages/About";
import Contact from "./pages/Contact";
import Dashboard from "./pages/Dashboard";
import CustomerAnalytics from "./pages/CustomerAnalytics";
import Alerts from "./pages/Alerts";

export default function App() {
  const [user, setUser] = useState(null);

  const handleLogout = () => setUser(null);
  const handleLogin = (userData) => setUser(userData);

  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen">
        <Navbar user={user} onLogout={handleLogout} />
        <div className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login onLogin={handleLogin} />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/customer-analytics" element={<CustomerAnalytics />} />
            <Route path="/alerts" element={<Alerts />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}