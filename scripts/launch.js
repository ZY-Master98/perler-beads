// One-click local launcher for the Next.js app. Uses only Node.js built-ins so
// it can install dependencies on a fresh checkout before starting the server.
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const host = '127.0.0.1';
const firstPort = 3000;
const lastPort = 3010;
const startupTimeoutMs = 120_000;
let server;
let stopping = false;

function npmProcess(args, options = {}) {
  if (process.platform === 'win32') {
    // cmd.exe is needed to run npm.cmd on Windows. All arguments are fixed by
    // this launcher; the only variable argument is a numeric local port.
    return spawn(process.env.ComSpec || 'cmd.exe', ['/d', '/s', '/c', `npm ${args.join(' ')}`], {
      cwd: root,
      stdio: 'inherit',
      ...options,
    });
  }
  return spawn('npm', args, { cwd: root, stdio: 'inherit', ...options });
}

function waitForExit(child) {
  return new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({ code, signal }));
  });
}

function stopServer() {
  if (stopping || !server || server.exitCode !== null || !server.pid) return;
  stopping = true;
  if (process.platform === 'win32') {
    // npm starts Next.js as a child; stop the whole process tree.
    spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  } else {
    server.kill('SIGTERM');
  }
}

process.once('SIGINT', () => {
  stopServer();
  process.exit(130);
});
process.once('SIGTERM', () => {
  stopServer();
  process.exit(143);
});
process.once('exit', stopServer);

function portAvailable(port) {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', (error) => {
      if (error.code === 'EADDRINUSE' || error.code === 'EACCES') resolve(false);
      else reject(error);
    });
    probe.listen(port, host, () => probe.close(() => resolve(true)));
  });
}

async function selectPort() {
  for (let port = firstPort; port <= lastPort; port += 1) {
    if (await portAvailable(port)) return port;
  }
  throw new Error(`No free local port found from ${firstPort} to ${lastPort}.`);
}

function isReady(url) {
  return new Promise((resolve) => {
    const request = http.get(url, (response) => {
      response.resume();
      resolve(response.statusCode >= 200 && response.statusCode < 400);
    });
    request.setTimeout(2_000, () => request.destroy());
    request.once('error', () => resolve(false));
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitUntilReady(url) {
  const deadline = Date.now() + startupTimeoutMs;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error('The development server exited before it was ready.');
    if (await isReady(url)) return;
    await delay(750);
  }
  throw new Error('The development server did not become ready within 2 minutes.');
}

async function openBrowser(url) {
  if (process.env.PERLER_LAUNCH_NO_BROWSER === '1') return;
  const command = process.platform === 'win32'
    ? 'powershell.exe'
    : (process.platform === 'darwin' ? 'open' : 'xdg-open');
  const args = process.platform === 'win32'
    ? ['-NoProfile', '-NonInteractive', '-Command', `Start-Process -FilePath '${url}' -WindowStyle Normal -ErrorAction Stop`]
    : [url];
  const browser = spawn(command, args, {
    stdio: process.platform === 'win32' ? 'inherit' : 'ignore',
    detached: process.platform !== 'win32',
  });
  try {
    const result = await waitForExit(browser);
    if (result.code !== 0) {
      console.warn(`Could not open the browser automatically (exit ${result.signal || result.code}). Open ${url} manually.`);
    }
  } catch (error) {
    console.warn(`Could not open the browser: ${error.message}. Open ${url} manually.`);
  }
}

async function main() {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 18 || (major === 18 && minor < 18)) {
    throw new Error('Node.js 18.18 or newer is required.');
  }

  if (!fs.existsSync(path.join(root, 'node_modules', 'next', 'dist', 'bin', 'next'))) {
    console.log('Installing dependencies with npm ci (first launch only)...');
    const install = npmProcess(['ci', '--no-audit', '--no-fund']);
    const result = await waitForExit(install);
    if (result.code !== 0) throw new Error('npm ci failed. Check the output above.');
  }

  const port = await selectPort();
  const url = `http://${host}:${port}`;
  console.log(`Starting Perler Beads at ${url} ...`);
  server = npmProcess(['run', 'dev', '--', '--hostname', host, '--port', String(port)]);
  const serverExit = waitForExit(server);

  try {
    await Promise.race([
      waitUntilReady(url),
      serverExit.then((result) => {
        throw new Error(`The development server exited before it was ready (${result.signal || result.code}).`);
      }),
    ]);
    console.log(`Ready: ${url}`);
    console.log('Press Ctrl+C or close this window to stop the server.');
    await openBrowser(url);
    const result = await serverExit;
    if (result.code !== 0) throw new Error(`The development server stopped (${result.signal || result.code}).`);
  } catch (error) {
    stopServer();
    throw error;
  }
}

main().catch((error) => {
  console.error(`Launcher error: ${error.message}`);
  process.exitCode = 1;
});
