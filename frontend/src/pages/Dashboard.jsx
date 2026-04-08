import { motion } from "framer-motion";

export default function Dashboard() {
  return (
    <div className="p-10 bg-gradient-to-br from-gray-900 to-black min-h-screen text-white">

      <h1 className="text-3xl mb-6">Dashboard</h1>

      <div className="dashboard-grid">

        {[1,2,3].map((item) => (
          <motion.div
            key={item}
            whileHover={{ scale: 1.05 }}
            className="p-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg"
          >
            <h2 className="text-xl">Card {item}</h2>
            <p>Real-time analytics</p>
          </motion.div>
        ))}

      </div>
    </div>
  );
}