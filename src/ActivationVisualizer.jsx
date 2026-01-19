import React, { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import { Info, Eye, Activity, Settings, Plus, X, RefreshCw, Filter, BookOpen } from "lucide-react";

const randomWeight = () => parseFloat((Math.random() * 2 - 1).toFixed(2));

const ActivationVisualizer = () => {
  const [input, setInput] = useState(0);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState([3, 2]); 
  const maxNeurons = 6;
  
  const [activations, setActivations] = useState(["tanh", "sigmoid"]);
  const availableActivations = ["relu", "sigmoid", "tanh", "linear"];

  const [viewMode, setViewMode] = useState("activity"); 
  const [showEditor, setShowEditor] = useState(false);
  
  // State for Graph Visibility Filtering
  const [hiddenLines, setHiddenLines] = useState(new Set());

  const activationFunctions = {
    relu: (x) => Math.max(0, x),
    sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    tanh: (x) => Math.tanh(x),
    linear: (x) => x,
  };

  // --- Initialization & State Management ---

  const initializeWeights = (layers) => {
    const W = [];
    const b = [];
    const layerSizes = [1, ...layers]; 
    for (let l = 1; l < layerSizes.length; l++) {
      const rows = layerSizes[l];
      const cols = layerSizes[l - 1];
      W.push(Array.from({ length: rows }, () => Array.from({ length: cols }, randomWeight)));
      b.push(Array.from({ length: rows }, randomWeight));
    }
    return [W, b];
  };

  const [[weights, biases], setParams] = useState(() => initializeWeights(neuronsPerLayer));

  const regenerateWeights = () => setParams(initializeWeights(neuronsPerLayer));

  // --- Handlers ---

  const addLayer = () => {
    if (neuronsPerLayer.length >= 5) return;
    const newLayers = [...neuronsPerLayer, 2];
    setNeuronsPerLayer(newLayers);
    setActivations([...activations, "relu"]);
    setParams(initializeWeights(newLayers));
  };

  const removeLayer = (idx) => {
    if (neuronsPerLayer.length <= 1) return;
    const newLayers = neuronsPerLayer.filter((_, i) => i !== idx);
    const newActivations = activations.filter((_, i) => i !== idx);
    setNeuronsPerLayer(newLayers);
    setActivations(newActivations);
    setParams(initializeWeights(newLayers));
  };

  const updateNeuronCount = (layerIdx, count) => {
    const newLayers = [...neuronsPerLayer];
    newLayers[layerIdx] = count;
    setNeuronsPerLayer(newLayers);
    setParams(initializeWeights(newLayers));
  };

  const updateWeight = (layerIdx, neuronIdx, inputIdx, val) => {
    const newWeights = weights.map(l => l.map(n => [...n])); 
    newWeights[layerIdx][neuronIdx][inputIdx] = val;
    setParams([newWeights, biases]);
  };

  const updateBias = (layerIdx, neuronIdx, val) => {
    const newBiases = biases.map(l => [...l]);
    newBiases[layerIdx][neuronIdx] = val;
    setParams([weights, newBiases]);
  };

  const toggleLineVisibility = (key) => {
    const newHidden = new Set(hiddenLines);
    if (newHidden.has(key)) {
      newHidden.delete(key);
    } else {
      newHidden.add(key);
    }
    setHiddenLines(newHidden);
  };

  // --- Math Core ---

  const forwardPass = (x) => {
    const layerOutputs = [];
    let inputVector = [x];
    for (let l = 0; l < neuronsPerLayer.length; l++) {
      const Wl = weights[l];
      const bl = biases[l];
      const activationName = activations[l] || "relu";
      const actFn = activationFunctions[activationName];
      
      const output = [];
      for (let i = 0; i < Wl.length; i++) {
        const z = Wl[i].reduce((sum, w, idx) => sum + w * inputVector[idx], 0) + bl[i];
        output.push({ z, a: actFn(z), inputs: [...inputVector] });
      }
      layerOutputs.push(output);
      inputVector = output.map((v) => v.a);
    }
    return layerOutputs;
  };

  const currentOutputs = useMemo(() => forwardPass(input), [input, weights, biases, neuronsPerLayer, activations]);

  const visualizationLayers = useMemo(() => {
    return [
      [{ a: input, isInput: true }], 
      ...currentOutputs
    ];
  }, [input, currentOutputs]);

  const graphData = useMemo(() => {
    const data = [];
    for (let x = -5; x <= 5; x += 0.2) {
      const outputs = forwardPass(x);
      const point = { x: parseFloat(x.toFixed(2)) };
      
      outputs.forEach((layer, lIdx) => {
        layer.forEach((neuron, nIdx) => {
          point[`L${lIdx + 1}_N${nIdx + 1}`] = parseFloat(neuron.a.toFixed(4));
        });
      });
      
      data.push(point);
    }
    return data;
  }, [weights, biases, activations, neuronsPerLayer]);

  // --- UI Components ---

  const ParameterScroller = ({ value, onChange, colorClass, min = -3, max = 3 }) => (
    <div className="flex flex-col items-center gap-1 w-full min-w-[140px]">
      <div className="flex justify-between w-full px-1">
         <span className="text-[10px] text-slate-500">{min}</span>
         <span className={`text-xs font-mono font-bold ${colorClass}`}>{value > 0 ? '+' : ''}{value.toFixed(2)}</span>
         <span className="text-[10px] text-slate-500">{max}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step="0.1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-3 bg-slate-700 rounded-lg appearance-none cursor-pointer hover:bg-slate-600 accent-slate-400"
      />
    </div>
  );

  const DescriptionCard = ({ title, children }) => (
    <div className="bg-slate-800/50 border border-slate-700 p-4 rounded-lg mb-4 text-sm text-slate-300">
      <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
        <BookOpen size={16} className="text-cyan-400"/> {title}
      </h4>
      {children}
    </div>
  );

  const svgWidth = 800;
  const svgHeight = 400;
  const layerSpacing = svgWidth / (visualizationLayers.length + 1);
  const neuronSpacing = 70;
  const neuronRadius = 24; 

  const getNodeColor = (val, mode, isInput = false) => {
    if (mode === 'weights') return { fill: "#1e293b", stroke: "#94a3b8" };
    const intensity = Math.min(Math.abs(val), 1);
    const minOpacity = 0.3; 
    const opacity = Math.max(intensity, minOpacity);

    if (val >= 0) {
      return { 
        fill: `rgba(34, 197, 94, ${opacity})`, 
        stroke: `rgba(34, 197, 94, 1)` 
      };
    } else {
      return { 
        fill: `rgba(249, 115, 22, ${opacity})`, 
        stroke: `rgba(249, 115, 22, 1)` 
      };
    }
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6 font-sans text-slate-200">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header & Main Actions */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white">DNN Laboratory</h1>
            <p className="text-slate-400 text-sm">Design, Build, and Inspect your Neural Network</p>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setShowEditor(!showEditor)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold transition-colors shadow-lg ${
                showEditor ? "bg-amber-600 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"
              }`}
            >
              <Settings size={16} /> {showEditor ? "Hide Parameters" : "Edit Parameters"}
            </button>
            <div className="w-px h-8 bg-slate-700 mx-2 hidden md:block"></div>
            <button
              onClick={() => setViewMode("activity")}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === "activity" ? "bg-green-600 text-white" : "bg-slate-800 border border-slate-600 hover:bg-slate-700"
              }`}
            >
              <Activity size={16} /> Activity
            </button>
            <button
              onClick={() => setViewMode("weights")}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                viewMode === "weights" ? "bg-blue-600 text-white" : "bg-slate-800 border border-slate-600 hover:bg-slate-700"
              }`}
            >
              <Eye size={16} /> Weights
            </button>
          </div>
        </div>

        {/* Global Controls */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-xl">
           <div className="flex flex-col md:flex-row gap-8">
              <div className="flex-1">
                <label className="text-white font-semibold flex items-center mb-4">
                  Input Signal (x)
                  <span className="ml-auto bg-slate-900 px-3 py-1 rounded font-mono text-cyan-400">{input.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={input}
                  onChange={(e) => setInput(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-2 font-mono">
                  <span>-5.0</span>
                  <span>0.0</span>
                  <span>+5.0</span>
                </div>
              </div>

              <div className="flex-[2] space-y-4">
                 <div className="flex justify-between items-center">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Architecture</h3>
                    <button onClick={addLayer} disabled={neuronsPerLayer.length >= 5} className="text-xs flex items-center gap-1 bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded disabled:opacity-50">
                      <Plus size={12}/> Add Layer
                    </button>
                 </div>
                 
                 <div className="flex flex-wrap gap-2">
                    {neuronsPerLayer.map((count, idx) => (
                      <div key={idx} className="bg-slate-900 p-3 rounded border border-slate-700 flex flex-col items-center gap-2 min-w-[100px] relative group">
                        {neuronsPerLayer.length > 1 && (
                          <button onClick={() => removeLayer(idx)} className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                            <X size={12}/>
                          </button>
                        )}
                        <span className="text-xs text-slate-500 font-mono">L{idx+1}</span>
                        <input 
                           type="range" min="1" max={maxNeurons} step="1" 
                           value={count} 
                           onChange={(e) => updateNeuronCount(idx, parseInt(e.target.value))}
                           className="w-20 accent-indigo-500 h-1 bg-slate-700 rounded appearance-none"
                        />
                        <span className="text-xs font-bold text-white">{count} Neurons</span>
                        <select
                          value={activations[idx] || "relu"}
                          onChange={(e) => {
                            const newActs = [...activations];
                            newActs[idx] = e.target.value;
                            setActivations(newActs);
                          }}
                          className="w-full text-xs bg-slate-800 text-slate-300 border border-slate-700 rounded px-1 py-1"
                        >
                          {availableActivations.map(fn => <option key={fn} value={fn}>{fn}</option>)}
                        </select>
                      </div>
                    ))}
                 </div>
              </div>
           </div>
        </div>

        {/* SECTION 1: Activation Landscape Graph */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 bg-slate-800 rounded-xl p-6 border border-slate-700 shadow h-[600px] flex flex-col">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-white mb-2">Activation Landscape</h2>
              <DescriptionCard title="What is this showing?">
                <p>
                  This graph plots the <strong>Response Curve</strong> of every neuron in the network.
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1 text-slate-400">
                  <li>The <strong>X-axis</strong> represents the range of possible Network Inputs (-5 to +5).</li>
                  <li>The <strong>Y-axis</strong> represents the Activation Output of a specific neuron.</li>
                  <li>The <span className="text-amber-400 font-bold">dotted vertical line</span> shows your <strong>Current Input</strong> value. Where this line intersects a curve tells you that neuron's current output.</li>
                </ul>
              </DescriptionCard>
            </div>
            
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={graphData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="x" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#f1f5f9" }}
                    itemStyle={{ fontSize: "12px" }}
                    labelStyle={{ color: "#94a3b8", marginBottom: "0.5rem" }}
                  />
                  <ReferenceLine x={input} stroke="#fbbf24" strokeWidth={2} strokeDasharray="4 4" label={{ value: "Current Input", fill: "#fbbf24", fontSize: 12, position: "top" }} />
                  
                  <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                  
                  {/* Individual Neuron Lines */}
                  {currentOutputs.map((layer, lIdx) =>
                    layer.map((_, nIdx) => {
                      const key = `L${lIdx + 1}_N${nIdx + 1}`;
                      const label = `L${lIdx + 1}N${nIdx + 1} (${activations[lIdx]})`;
                      if (hiddenLines.has(key)) return null;

                      return (
                        <Line
                          key={key}
                          name={label}
                          type="monotone"
                          dataKey={key}
                          stroke={`hsl(${(lIdx * 80 + nIdx * 40) % 360}, 70%, 50%)`}
                          strokeWidth={3}
                          dot={false}
                        />
                      );
                    })
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Graph Filters */}
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 shadow h-fit">
             <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
               <Filter size={16} className="text-slate-400" /> Filter Lines
             </h3>
             <p className="text-xs text-slate-400 mb-4">Click to hide/show specific neurons in the graph to reduce clutter.</p>
             <div className="space-y-4">
               {currentOutputs.map((layer, lIdx) => (
                 <div key={lIdx} className="space-y-2">
                   <div className="text-xs font-bold text-slate-500 uppercase">Layer {lIdx+1}</div>
                   <div className="flex flex-wrap gap-2">
                     {layer.map((_, nIdx) => {
                       const key = `L${lIdx + 1}_N${nIdx + 1}`;
                       const isHidden = hiddenLines.has(key);
                       const color = `hsl(${(lIdx * 80 + nIdx * 40) % 360}, 70%, 50%)`;
                       return (
                         <button
                           key={key}
                           onClick={() => toggleLineVisibility(key)}
                           className={`px-3 py-1 rounded text-xs font-bold border transition-all ${
                             isHidden 
                               ? "bg-slate-900 text-slate-500 border-slate-700 opacity-50" 
                               : "bg-slate-900 text-white border-slate-600"
                           }`}
                           style={{ borderColor: isHidden ? undefined : color }}
                         >
                            <span style={{ color: isHidden ? undefined : color }}>●</span> N{nIdx+1}
                         </button>
                       );
                     })}
                   </div>
                 </div>
               ))}
               {currentOutputs.flat().length > 6 && (
                 <button 
                   onClick={() => setHiddenLines(new Set())}
                   className="text-xs text-blue-400 hover:text-blue-300 mt-4 underline"
                 >
                   Show All
                 </button>
               )}
             </div>
          </div>
        </div>

        {/* SECTION 2: Manual Parameter Editor */}
        {showEditor && (
          <div className="bg-slate-800 rounded-xl border border-amber-600/50 shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-4 duration-300">
            <div className="bg-slate-900/50 p-4 border-b border-slate-700 flex justify-between items-center">
               <div>
                  <h3 className="text-lg font-bold text-amber-500 flex items-center gap-2">
                    <Settings size={18}/> Manual Parameter Configuration
                  </h3>
                  <p className="text-slate-400 text-xs">Directly edit the Weight Matrix (W) and Bias Vector (b) for each layer.</p>
               </div>
               <button onClick={regenerateWeights} className="flex items-center gap-2 text-xs bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded text-white transition-colors">
                  <RefreshCw size={14}/> Randomize All
               </button>
            </div>
            
            <div className="p-6 grid grid-cols-1 gap-8">
              {weights.map((layerWeights, lIdx) => (
                <div key={lIdx} className="bg-slate-900 p-6 rounded-lg border border-slate-700">
                  <h4 className="font-mono text-lg text-slate-300 mb-4 border-b border-slate-700 pb-2 flex justify-between">
                    <span>Layer {lIdx + 1} Parameters</span>
                  </h4>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm border-separate border-spacing-y-4">
                      <thead>
                        <tr>
                          <th className="text-left text-slate-500 p-2 w-24">Neuron</th>
                          <th className="text-center text-amber-500 p-2 border-r border-slate-700 w-48">Bias (b)</th>
                          {layerWeights[0].map((_, i) => (
                            <th key={i} className="text-center text-blue-400 p-2 min-w-[180px]">
                              Weight from {lIdx === 0 ? "Input" : `L${lIdx}N${i+1}`}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {layerWeights.map((neuronWeights, nIdx) => (
                          <tr key={nIdx} className="bg-slate-800/40 hover:bg-slate-800/80 transition-colors">
                            <td className="font-bold text-slate-300 p-3 rounded-l">L{lIdx+1} Neuron {nIdx + 1}</td>
                            <td className="p-3 border-r border-slate-700 flex justify-center">
                              <ParameterScroller 
                                value={biases[lIdx][nIdx]} 
                                onChange={(val) => updateBias(lIdx, nIdx, val)}
                                colorClass="text-amber-300"
                              />
                            </td>
                            {neuronWeights.map((w, wIdx) => (
                              <td key={wIdx} className="p-3">
                                <div className="flex justify-center">
                                  <ParameterScroller 
                                    value={w} 
                                    onChange={(val) => updateWeight(lIdx, nIdx, wIdx, val)}
                                    colorClass="text-blue-300"
                                  />
                                </div>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SECTION 3: Visualizer Diagram */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow overflow-hidden relative">
          <div className="mb-4">
             <h2 className="text-lg font-semibold text-white mb-2">Network Diagram</h2>
             <DescriptionCard title="Understanding the Diagram">
                <p>
                  This illustrates the actual architecture.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                   <ul className="list-disc list-inside text-slate-400 space-y-1">
                      <li><strong>Circles (Neurons):</strong> Fill opacity represents activation strength. <span className="text-green-400">Green</span> is positive, <span className="text-orange-400">Orange</span> is negative.</li>
                      <li><strong>Lines (Weights):</strong> Thickness represents the weight's magnitude.</li>
                   </ul>
                   <div className="text-xs text-slate-500 bg-slate-900 p-2 rounded">
                      <strong>Tip:</strong> Toggle the "Weights" view mode (top right) to see the fixed structure (Blue=Positive connection, Red=Negative connection) instead of the active signal flow.
                   </div>
                </div>
             </DescriptionCard>
          </div>
          
          <div className="flex justify-center">
            <svg width={svgWidth} height={svgHeight} className="overflow-visible">
              <defs>
                <marker id="arrow" markerWidth="6" markerHeight="6" refX="16" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#64748b" />
                </marker>
              </defs>
              
              {/* Lines */}
              {visualizationLayers.map((layer, lIdx) =>
                layer.map((neuron, nIdx) => {
                  const x = layerSpacing * (lIdx + 1);
                  const y = svgHeight / 2 - ((layer.length - 1) * neuronSpacing) / 2 + nIdx * neuronSpacing;

                  if (lIdx < visualizationLayers.length - 1) {
                    return visualizationLayers[lIdx + 1].map((nextNeuron, nextIdx) => {
                      const nx = layerSpacing * (lIdx + 2);
                      const ny = svgHeight / 2 - ((visualizationLayers[lIdx + 1].length - 1) * neuronSpacing) / 2 + nextIdx * neuronSpacing;

                      let strokeColor, strokeWidth, opacity;

                      if (viewMode === "weights") {
                        const wVal = weights[lIdx][nextIdx][nIdx];
                        const intensity = Math.min(Math.abs(wVal), 1);
                        strokeColor = wVal >= 0 ? `rgba(59, 130, 246, ${intensity})` : `rgba(239, 68, 68, ${intensity})`;
                        strokeWidth = Math.max(1, Math.abs(wVal) * 3);
                        opacity = 0.6;
                      } else {
                        const val = neuron.a;
                        const styles = getNodeColor(val, 'activity');
                        strokeColor = styles.stroke;
                        strokeWidth = 2;
                        opacity = styles.fill.split(',')[3].replace(')', ''); 
                      }

                      return (
                        <line
                          key={`link-${lIdx}-${nIdx}-${nextIdx}`}
                          x1={x} y1={y} x2={nx} y2={ny}
                          stroke={strokeColor}
                          strokeWidth={strokeWidth}
                          strokeOpacity={opacity}
                          strokeLinecap="round"
                          style={{ transition: "stroke 0.2s" }}
                        />
                      );
                    });
                  }
                  return null;
                })
              )}

              {/* Nodes */}
              {visualizationLayers.map((layer, lIdx) =>
                layer.map((neuron, nIdx) => {
                  const x = layerSpacing * (lIdx + 1);
                  const y = svgHeight / 2 - ((layer.length - 1) * neuronSpacing) / 2 + nIdx * neuronSpacing;
                  
                  const styles = getNodeColor(neuron.a, viewMode);
                  
                  let label = `L${lIdx}N${nIdx+1}`;
                  if (neuron.isInput) {
                    label = "Input";
                  } else if (lIdx === visualizationLayers.length - 1) {
                    label = "Output";
                  }

                  return (
                    <g key={`node-${lIdx}-${nIdx}`}>
                      <circle
                        cx={x} cy={y} r={neuronRadius}
                        fill={styles.fill}
                        stroke={styles.stroke}
                        strokeWidth={2}
                        className="transition-colors duration-300"
                      />
                      <text x={x} y={y} dy={4} textAnchor="middle" fontSize="10" fill="white" fontWeight="bold" pointerEvents="none" style={{textShadow: '0 1px 2px rgba(0,0,0,0.8)'}}>
                        {neuron.a.toFixed(1)}
                      </text>
                      <text x={x} y={y - 30} textAnchor="middle" fontSize="10" fill="#64748b" fontWeight="bold">
                        {label}
                      </text>
                    </g>
                  );
                })
              )}
            </svg>
          </div>
        </div>

        {/* SECTION 4: Trace Table */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow overflow-x-auto">
          <div className="mb-4">
             <h2 className="text-xl font-semibold text-white mb-2">Computation Trace</h2>
             <DescriptionCard title="The Math Behind the Magic">
                <p>
                  Every neuron performs a simple 2-step calculation:
                </p>
                <ol className="list-decimal list-inside mt-2 space-y-1 text-slate-400 font-mono text-xs md:text-sm">
                  <li>
                    <strong>Linear Step (z):</strong> Multiply inputs by weights and add bias. <br/>
                    <span className="text-yellow-400">z = (w₁x₁ + w₂x₂ + ...) + b</span>
                  </li>
                  <li>
                    <strong>Non-Linear Step (a):</strong> Apply the selected activation function. <br/>
                    <span className="text-green-400">a = f(z)</span>
                    <span className="text-slate-500 block text-[10px] ml-4 italic">(where 'f' is ReLU, Sigmoid, etc.)</span>
                  </li>
                </ol>
             </DescriptionCard>
          </div>
          <table className="w-full text-sm border-collapse">
            <thead className="bg-slate-800 shadow-sm">
              <tr className="text-left text-slate-400 border-b border-slate-700">
                <th className="py-2 px-2">Neuron</th>
                <th className="py-2 px-2">Activation Function</th>
                <th className="py-2 px-2">Bias (b)</th>
                <th className="py-2 px-2">Incoming Weights (w)</th>
                <th className="py-2 px-2">Net Input (z)</th>
                <th className="py-2 px-2">Output (a)</th>
              </tr>
            </thead>
            <tbody className="font-mono text-slate-300">
              {currentOutputs.map((layer, lIdx) =>
                layer.map((neuron, nIdx) => (
                  <tr key={`${lIdx}-${nIdx}`} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-2 px-2 text-slate-400">Layer {lIdx + 1} - Neuron {nIdx + 1}</td>
                    <td className="py-2 px-2 text-slate-400 uppercase tracking-wider">{activations[lIdx]}</td>
                    <td className="py-2 px-2 text-amber-500">{biases[lIdx][nIdx].toFixed(2)}</td>
                    <td className="py-2 px-2 text-blue-400 text-xs">
                      [{weights[lIdx][nIdx].map(w => w.toFixed(2)).join(", ")}]
                    </td>
                    <td className="py-2 px-2 text-yellow-500">{neuron.z.toFixed(3)}</td>
                    <td className={`py-2 px-2 font-bold ${neuron.a >= 0 ? 'text-green-400' : 'text-orange-400'}`}>{neuron.a.toFixed(3)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

      </div>
    </div>
  );
};

export default ActivationVisualizer;