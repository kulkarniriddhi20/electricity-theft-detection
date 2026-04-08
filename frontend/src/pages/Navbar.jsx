import React from "react";
import { NavLink } from "react-router-dom";

const Navbar = ({ user, onLogout }) => {
  return (
    <nav className="bg-blue-600 text-white px-6 py-4 flex justify-between items-center shadow-md">
      <div className="text-xl font-bold">
        <NavLink to="/">Electricity Theft Detection</NavLink>
      </div>

      <ul className="flex items-center space-x-6">
        <li>
          <NavLink 
            to="/" 
            className={({ isActive }) => isActive ? "underline" : "hover:text-gray-200"}
          >
            Home
          </NavLink>
        </li>
        <li>
          <NavLink 
            to="/about" 
            className={({ isActive }) => isActive ? "underline" : "hover:text-gray-200"}
          >
            About
          </NavLink>
        </li>
        <li>
          <NavLink 
            to="/contact" 
            className={({ isActive }) => isActive ? "underline" : "hover:text-gray-200"}
          >
            Contact Us
          </NavLink>
        </li>
        {!user && (
          <li>
            <NavLink 
              to="/login" 
              className={({ isActive }) => isActive ? "underline" : "hover:text-gray-200"}
            >
              Login
            </NavLink>
          </li>
        )}
        {user && (
          <>
            <li className="font-medium">Hi, {user.name}</li>
            <li>
              <button onClick={onLogout} className="hover:text-gray-200">
                Logout
              </button>
            </li>
          </>
        )}
      </ul>
    </nav>
  );
};

export default Navbar;