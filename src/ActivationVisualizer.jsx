import React, { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Info } from "lucide-react";

const randomWeight = () => parseFloat((Math.random() * 2 - 1).toFixed(2));

const ActivationVisualizer = () => {
  const [input, setInput] = useState(0);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState([3, 3, 2]);
  const maxNeurons = 8;
  const [activations] = useState(["relu", "sigmoid", "tanh"]);

  // Activation functions
  const relu = (x) => Math.max(0, x);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const tanh = (x) => Math.tanh(x);
  const activationFunctions = { relu, sigmoid, tanh };

  // Initialize weights and biases
  const initializeWeights = () => {
    const W = [];
    const b = [];
    const layerSizes = [1, ...neuronsPerLayer];
    for (let l = 1; l < layerSizes.length; l++) {
      const rows = layerSizes[l];
      const cols = layerSizes[l - 1];
      W.push(Array.from({ length: rows }, () => Array.from({ length: cols }, randomWeight)));
      b.push(Array.from({ length: rows }, randomWeight));
    }
    return [W, b];
  };

  const [[weights, biases], setParams] = useState(() => initializeWeights());
  const regenerateWeights = () => setParams(initializeWeights());

  // Forward pass
  const forwardPass = (x) => {
    const layerOutputs = [];
    let inputVector = [x];
    for (let l = 0; l < neuronsPerLayer.length; l++) {
      const Wl = weights[l];
      const bl = biases[l];
      const activation = activations[l];
      const output = [];
      for (let i = 0; i < Wl.length; i++) {
        const z = Wl[i].reduce((sum, w, idx) => sum + w * inputVector[idx], 0) + bl[i];
        output.push({ z, a: activationFunctions[activation](z), inputs: [...inputVector] });
      }
      layerOutputs.push(output);
      inputVector = output.map((v) => v.a);
    }
    return layerOutputs;
  };

  const currentOutputs = useMemo(() => forwardPass(input), [input, weights, biases, neuronsPerLayer, activations]);

  // Graph data
  const graphData = useMemo(() => {
    const data = [];
    for (let x = -5; x <= 5; x += 0.1) {
      const outputs = forwardPass(x);
      const point = { x: parseFloat(x.toFixed(2)) };
      outputs.forEach((layer, lIdx) =>
        layer.forEach((neuron, nIdx) => {
          point[`L${lIdx + 1}_N${nIdx + 1}`] = parseFloat(neuron.a.toFixed(4));
        })
      );
      data.push(point);
    }
    return data;
  }, [weights, biases, activations, neuronsPerLayer]);

  const InfoTooltip = ({ text }) => (
    <div className="group relative inline-block ml-1">
      <Info size={14} className="text-slate-400 cursor-help" />
      <div className="invisible group-hover:visible absolute z-10 w-64 p-2 mt-1 text-xs bg-slate-900 text-slate-200 rounded shadow-lg border border-slate-700 left-0">
        {text}
      </div>
    </div>
  );

  // Layout for SVG network diagram
  const svgWidth = 800;
  const svgHeight = 300;
  const layerSpacing = svgWidth / (neuronsPerLayer.length + 1);
  const neuronSpacing = 60;
  const neuronRadius = 15;

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">Multi-Neuron Forward Pass Visualizer</h1>
          <p className="text-slate-300">
            Move the input slider to see signals propagate through the network.
          </p>
        </div>

        {/* Controls */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 shadow">
            <label className="text-slate-300 font-semibold block mb-1">
              Input Value: {input.toFixed(2)}
              <InfoTooltip text="Adjust the input signal to see its effect through all neurons." />
            </label>
            <input
              type="range"
              min="-5"
              max="5"
              step="0.1"
              value={input}
              onChange={(e) => setInput(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          {neuronsPerLayer.map((count, idx) => (
            <div key={idx} className="bg-slate-800 rounded-xl p-4 border border-slate-700 shadow">
              <label className="text-slate-300 font-semibold block mb-1">
                Layer {idx + 1} Neurons: {count}
                <InfoTooltip text="Number of neurons in this layer." />
              </label>
              <input
                type="range"
                min="1"
                max={maxNeurons}
                step="1"
                value={count}
                onChange={(e) => {
                  const newCounts = [...neuronsPerLayer];
                  newCounts[idx] = parseInt(e.target.value);
                  setNeuronsPerLayer(newCounts);
                  regenerateWeights();
                }}
                className="w-full"
              />
            </div>
          ))}

          <button
            onClick={regenerateWeights}
            className="bg-blue-600 text-white rounded-xl py-2 px-4 hover:bg-blue-500 self-end shadow"
          >
            Regenerate Weights
          </button>
        </div>

        {/* Network Diagram */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow overflow-x-auto">
          <h2 className="text-xl font-semibold text-white mb-4">Network Diagram</h2>
          <svg width={svgWidth} height={svgHeight}>
            {currentOutputs.map((layer, lIdx) =>
              layer.map((neuron, nIdx) => {
                const x = layerSpacing * (lIdx + 1);
                const y =
                  svgHeight / 2 -
                  ((layer.length - 1) * neuronSpacing) / 2 +
                  nIdx * neuronSpacing;

                // Draw lines to next layer
                if (lIdx < currentOutputs.length - 1) {
                  currentOutputs[lIdx + 1].forEach((nextNeuron, nextIdx) => {
                    const nx = layerSpacing * (lIdx + 2);
                    const ny =
                      svgHeight / 2 -
                      ((currentOutputs[lIdx + 1].length - 1) * neuronSpacing) / 2 +
                      nextIdx * neuronSpacing;
                    const intensity = neuron.a; // Use activation to "light up"
                    const color = `rgba(34, 197, 94, ${intensity})`; // green
                    // Draw animated line
                    const dashOffset = 10 - intensity * 10;
                    return (
                      <line
                        key={`line-${lIdx}-${nIdx}-${nextIdx}`}
                        x1={x}
                        y1={y}
                        x2={nx}
                        y2={ny}
                        stroke={color}
                        strokeWidth={3}
                        strokeLinecap="round"
                        strokeDasharray="4 4"
                        style={{ transition: "stroke 0.3s, stroke-dashoffset 0.3s" }}
                      />
                    );
                  });
                }

                return (
                  <circle
                    key={`neuron-${lIdx}-${nIdx}`}
                    cx={x}
                    cy={y}
                    r={neuronRadius}
                    fill="rgb(100 116 139)"
                    stroke="white"
                    strokeWidth={2}
                  />
                );
              })
            )}
          </svg>
        </div>

        {/* Forward Pass Table */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow overflow-x-auto">
          <h2 className="text-xl font-semibold text-white mb-4">Forward Pass Table</h2>
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-2 px-3 text-slate-300">Layer</th>
                <th className="text-left py-2 px-3 text-slate-300">Neuron</th>
                <th className="text-left py-2 px-3 text-slate-300">Inputs</th>
                <th className="text-left py-2 px-3 text-slate-300">z</th>
                <th className="text-left py-2 px-3 text-slate-300">Activation</th>
                <th className="text-left py-2 px-3 text-slate-300">Output (a)</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {currentOutputs.map((layer, lIdx) =>
                layer.map((neuron, nIdx) => (
                  <tr
                    key={`L${lIdx}_N${nIdx}`}
                    className={`border-b border-slate-700`}
                  >
                    <td className="py-2 px-3 text-white">{lIdx + 1}</td>
                    <td className="py-2 px-3 text-white">{nIdx + 1}</td>
                    <td className="py-2 px-3 text-slate-300">
                      [{neuron.inputs.map((v) => v.toFixed(2)).join(", ")}]
                    </td>
                    <td className="py-2 px-3 text-yellow-300">{neuron.z.toFixed(3)}</td>
                    <td className="py-2 px-3 text-slate-300">{activations[lIdx]}</td>
                    <td className="py-2 px-3 text-green-300 font-bold">{neuron.a.toFixed(3)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Output Curves */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow">
          <h2 className="text-xl font-semibold text-white mb-4">Neuron Output Curves</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={graphData} margin={{ top: 5, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="x" stroke="#94a3b8" label={{ value: "Input", position: "insideBottom", fill: "#94a3b8" }} />
              <YAxis stroke="#94a3b8" label={{ value: "Neuron Output", angle: -90, position: "insideLeft", fill: "#94a3b8" }} />
              <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569", borderRadius: "8px" }} />
              <Legend />
              {currentOutputs.map((layer, lIdx) =>
                layer.map((_, nIdx) => {
                  const key = `L${lIdx + 1}_N${nIdx + 1}`;
                  return (
                    <Line key={key} type="monotone" dataKey={key} stroke={`hsl(${(lIdx * 60 + nIdx * 30) % 360}, 70%, 40%)`} strokeWidth={2} dot={false} />
                  );
                })
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default ActivationVisualizer;
