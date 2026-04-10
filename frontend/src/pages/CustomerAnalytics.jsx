import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip
} from "recharts";

export default function CustomerAnalytics() {

  const [selectedId, setSelectedId] = useState("");

  // ✅ Get data
  const data = JSON.parse(localStorage.getItem("customerData")) || [];

  // ❌ No data case
  if (!data.length) {
    return (
      <div className="p-10 text-white">
        <h2>No Data Available</h2>
        <p>Please upload file first</p>
      </div>
    );
  }

  // ✅ Selected customer
  const selectedCustomer = data.find(
    (c) => c.customer_id === Number(selectedId)
  );

  // ✅ Graph data
  const chartData = selectedCustomer
    ? [
        { name: "Daily", value: selectedCustomer.daily_mean },
        { name: "Night", value: selectedCustomer.night_ratio },
        { name: "Weekend", value: selectedCustomer.weekend_ratio },
        { name: "Variance", value: selectedCustomer.variance }
      ]
    : [];

  // ✅ Risk
  const risk =
    selectedCustomer?.probability > 0.8
      ? "High Risk ⚠"
      : selectedCustomer?.probability > 0.5
      ? "Medium Risk"
      : "Low Risk";

  return (
    <div className="p-10 bg-gray-900 text-white min-h-screen">

      <h1 className="text-3xl mb-6">Customer Analytics</h1>

      {/* Dropdown */}
      <select
        onChange={(e) => setSelectedId(e.target.value)}
        className="p-2 text-black rounded mb-6"
      >
        <option value="">Select Customer</option>

        {data.map((d) => (
          <option key={d.customer_id} value={d.customer_id}>
            {d.customer_id}
          </option>
        ))}
      </select>

      {/* Details */}
      {selectedCustomer && (
        <div className="mb-6 p-4 bg-white/10 rounded">
          <p>Customer ID: {selectedCustomer.customer_id}</p>
          <p>Prediction: {selectedCustomer.prediction}</p>
          <p>Probability: {selectedCustomer.probability.toFixed(3)}</p>
          <p>Risk: {risk}</p>
        </div>
      )}

      {/* Graph */}
      {selectedCustomer && (
        <BarChart width={400} height={250} data={chartData}>
          <XAxis dataKey="name" stroke="#fff"/>
          <YAxis stroke="#fff"/>
          <Tooltip />
          <Bar dataKey="value" fill="#00C49F" />
        </BarChart>
      )}

    </div>
  );
}