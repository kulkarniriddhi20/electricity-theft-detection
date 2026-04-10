import { motion } from "framer-motion";

export default function Alerts() {

  // Get stored data
  const data = JSON.parse(localStorage.getItem("customerData")) || [];

  // Filter high-risk customers
  const highRisk = data.filter(d => d.probability > 0.8);

  // No data case
  if (!data.length) {
    return (
      <div className="p-10 text-white bg-gray-900 min-h-screen">
        <h2>No Data Available</h2>
        <p>Please upload customer file first</p>
      </div>
    );
  }

  return (
    <div className="p-10 bg-gradient-to-br from-black to-gray-900 text-white min-h-screen">

      <h1 className="text-3xl mb-6">⚡ High-Risk Alerts</h1>

      {/* No high-risk */}
      {highRisk.length === 0 && (
        <p>No high-risk customers found ✅</p>
      )}

      {/* Alerts Cards */}
      <div className="grid grid-cols-3 gap-6">

        {highRisk.map((d, i) => (
          <motion.div
            key={i}
            whileHover={{ scale: 1.05 }}
            className="p-5 bg-red-500/20 border border-red-500 rounded-xl shadow-lg"
          >
            <h2 className="text-xl">Customer ID: {d.customer_id}</h2>
            <p>Prediction: {d.prediction}</p>
            <p>Probability: {d.probability.toFixed(3)}</p>

            <p className="text-red-400 font-bold mt-2">
              🚨 HIGH RISK
            </p>
          </motion.div>
        ))}

      </div>
    </div>
  );
}