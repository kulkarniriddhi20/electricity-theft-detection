import { useLocation } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell
} from "recharts";
import { motion } from "framer-motion";

export default function Analytics() {

  const location = useLocation();
  const data = location.state || [];

  // 🔥 calculations
  const total = data.length;
  const theft = data.filter(d => d.prediction === 1).length;
  const normal = data.filter(d => d.prediction === 0).length;

  const avgProb = (
    data.reduce((sum, d) => sum + d.probability, 0) / total || 0
  ).toFixed(2);

  const chartData = [
    { name: "Normal", value: normal },
    { name: "Theft", value: theft }
  ];

  // ✅ 📥 DOWNLOAD FUNCTION (ADD HERE)
  const downloadCSV = () => {

    if (!data.length) {
      alert("No data available");
      return;
    }

    const headers = ["customer_id", "prediction", "probability"];

    const rows = data.map(d =>
      [d.customer_id, d.prediction, d.probability].join(",")
    );

    const csvContent =
      headers.join(",") + "\n" + rows.join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");

    a.href = url;
    a.download = "prediction_report.csv";
    a.click();

    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="p-10 bg-gradient-to-br from-black to-gray-900 text-white min-h-screen">

      <h1 className="text-3xl mb-6">📊 Dashboard</h1>

      {/* ✅ DOWNLOAD BUTTON */}
      <button
        onClick={downloadCSV}
        className="bg-blue-600 px-4 py-2 rounded mb-6 hover:bg-blue-700"
      >
        ⬇ Download Report
      </button>

      {/* 🔥 CARDS */}
      <div className="grid grid-cols-4 gap-6 mb-8">

        {[
          { title: "Total Users", value: total },
          { title: "Theft Cases", value: theft },
          { title: "Normal Users", value: normal },
          { title: "Avg Risk", value: avgProb }
        ].map((card, i) => (
          <motion.div key={i}
            whileHover={{ scale: 1.05 }}
            className="p-5 bg-white/10 rounded-xl backdrop-blur-lg"
          >
            <h2>{card.title}</h2>
            <p className="text-2xl">{card.value}</p>
          </motion.div>
        ))}
      </div>

      {/* 🔥 CHARTS */}
      <div className="grid grid-cols-2 gap-8">

        <div className="bg-white/10 p-6 rounded-xl">
          <h2>Theft Distribution</h2>

          <BarChart width={300} height={250} data={chartData}>
            <XAxis dataKey="name" stroke="#fff"/>
            <YAxis stroke="#fff"/>
            <Tooltip />
            <Bar dataKey="value" fill="#4CAF50" />
          </BarChart>
        </div>

        <div className="bg-white/10 p-6 rounded-xl">
          <h2>Ratio</h2>

          <PieChart width={300} height={250}>
            <Pie data={chartData} dataKey="value" outerRadius={80}>
              <Cell fill="#00C49F" />
              <Cell fill="#FF4C4C" />
            </Pie>
          </PieChart>
        </div>

      </div>

      {/* 🔥 TABLE */}
      <div className="mt-10">
        <h2 className="mb-3">Prediction Table</h2>

        <table className="w-full text-left border">
          <thead>
            <tr>
              <th>ID</th>
              <th>Prediction</th>
              <th>Probability</th>
            </tr>
          </thead>

          <tbody>
            {data.map((d, i) => (
              <tr key={i} className="border-t">
                <td>{d.customer_id}</td>
                <td>{d.prediction}</td>
                <td>{d.probability.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}