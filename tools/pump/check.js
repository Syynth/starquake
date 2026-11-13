#!/usr/bin/env node
// Dry-runs a filled pump script with stubbed agents, so a wave's script is known to parse and
// complete before it is launched. Usage: node tools/pump/check.js <filled-script.js>
const fs = require("fs");
const path = process.argv[2];
if (!path) { console.error("usage: node tools/pump/check.js <filled-script.js>"); process.exit(2); }
const body = fs.readFileSync(path, "utf8").replace("export const meta", "const meta");
const PREFIX = "const log=()=>{}, phase=()=>{};\nlet n=0; const agent=async(p,o)=>{ if(o.schema && o.label.startsWith(\"build\")) return {ok:true,issue:1,pr:\"https://x/pull/9\",gateGreen:true,reachability:\"r\",summary:\"s\"}; if(o.label.startsWith(\"review\")) return {pr:\"https://x/pull/9\",decision:\"approve\",findings:[],scopeGaps:[\"g1\"]}; if(o.label.startsWith(\"break\")) return {pr:\"https://x/pull/9\",decision:\"changes\",findings:[\"mutant survived\"],scopeGaps:[]}; if(o.label.startsWith(\"fix\")||o.label.startsWith(\"merge\")) return {pr:\"9\",merged:true,detail:o.label}; if(o.label===\"lessons\") return {lessons:[\"l\"]}; return {assessment:\"scope-held\",recommendation:\"scope-held\",proposal:\"p\"}; };\nconst parallel=async(ts)=>Promise.all(ts.map(t=>t()));\nconst pipeline=async(items,...fns)=>Promise.all(items.map(async(it)=>{ let v=await fns[0](it); for(const f of fns.slice(1)) v=await f(v,it); return v; }));\n(async()=>{\n";
const SUFFIX = "})().then(r=>console.log(\"ran ok\", JSON.stringify(r.counts)));";
const harness = PREFIX + body + "\n" + SUFFIX;
new Function(harness)();
