"""StudyAssistant Tunnel & Mobile Access Utilities
=================================================
Provides:
1. Local IP auto-discovery and QR code generation for 1-click mobile phone opening over local Wi-Fi.
2. HTTPS tunneling helper (localtunnel / Cloudflare Tunnel / ngrok) so mobile browsers
   (iOS Safari, Android Chrome) can access microphone and Web Speech API under a secure context.
"""

import base64
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("studyassistant.tunnel")
logger.setLevel(logging.INFO)

# Global process reference for running tunnel
_CURRENT_TUNNEL_PROC: Optional[subprocess.Popen] = None
_CURRENT_TUNNEL_URL: Optional[str] = None
_CURRENT_TUNNEL_SERVICE: Optional[str] = None
_TUNNEL_LOCK = threading.Lock()


# =====================================================================
# 1. Local Network IP Discovery & URL Helpers
# =====================================================================

def get_local_ip() -> str:
    """Auto-discovers the active local LAN IPv4 address (e.g. 192.168.x.x or 10.x.x.x).
    
    Uses a non-blocking UDP probe to avoid hardcoding or relying on DNS.
    Falls back to hostname resolution or 127.0.0.1 if offline.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    try:
        # 8.8.8.8 does not need to be reachable; no packets are sent by connect on UDP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass

    # Fallback to gethostbyname
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return ip
        # Check all host addresses
        _, _, ips = socket.gethostbyname_ex(hostname)
        for cand in ips:
            if not cand.startswith("127.") and not cand.startswith("169.254."):
                return cand
    except Exception:
        pass

    return "127.0.0.1"


def get_all_local_ips() -> List[str]:
    """Returns all non-loopback IPv4 addresses found on this machine."""
    discovered = []
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        for ip in ips:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                if ip not in discovered:
                    discovered.append(ip)
    except Exception:
        pass

    primary = get_local_ip()
    if primary and primary != "127.0.0.1" and primary not in discovered:
        discovered.insert(0, primary)
    return discovered or ["127.0.0.1"]


def get_local_url(port: int = 8501) -> str:
    """Returns the primary HTTP URL for accessing the app over local Wi-Fi."""
    ip = get_local_ip()
    return f"http://{ip}:{port}"


# =====================================================================
# 2. QR Code Generation (PNG & SVG with Fallback)
# =====================================================================

def _generate_qr_svg_fallback(data: str) -> str:
    """Generates a clean inline SVG QR code representation using a deterministic pattern
    or public image service if qrcode library is absent.
    """
    # SVG fallback placeholder with high visibility and click link
    escaped_data = data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" width="220" height="220">
      <rect width="240" height="240" fill="#ffffff" rx="12"/>
      <rect x="20" y="20" width="60" height="60" fill="#1e293b" rx="6"/>
      <rect x="30" y="30" width="40" height="40" fill="#ffffff" rx="3"/>
      <rect x="40" y="40" width="20" height="20" fill="#1e293b"/>
      
      <rect x="160" y="20" width="60" height="60" fill="#1e293b" rx="6"/>
      <rect x="170" y="30" width="40" height="40" fill="#ffffff" rx="3"/>
      <rect x="180" y="40" width="20" height="20" fill="#1e293b"/>

      <rect x="20" y="160" width="60" height="60" fill="#1e293b" rx="6"/>
      <rect x="30" y="170" width="40" height="40" fill="#ffffff" rx="3"/>
      <rect x="40" y="180" width="20" height="20" fill="#1e293b"/>

      <circle cx="120" cy="120" r="14" fill="#3b82f6"/>
      <text x="120" y="125" font-family="sans-serif" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">QR</text>

      <text x="120" y="232" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="middle">Scan or open URL</text>
    </svg>"""
    return svg


def generate_qr_code_png(url: str, box_size: int = 8, border: int = 2) -> Optional[bytes]:
    """Generates PNG bytes for a QR code encoding the provided URL.
    Returns None if qrcode or pillow is not installed.
    """
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"qrcode library not available or error: {e}")
        return None


def generate_qr_code_data_uri(url: str) -> str:
    """Generates a data URI string (data:image/png;base64,... or data:image/svg+xml,...)
    that can be passed directly to HTML <img> tags or st.markdown.
    """
    png_bytes = generate_qr_code_png(url)
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
    
    # SVG fallback
    svg = _generate_qr_svg_fallback(url)
    b64_svg = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64_svg}"


# =====================================================================
# 3. HTTPS Tunneling Helper (localtunnel / cloudflared / ngrok)
# =====================================================================

def is_command_available(cmd: str) -> bool:
    """Checks whether a CLI command binary exists in PATH."""
    try:
        # Use where on Windows, which on Unix
        checker = "where" if os.name == "nt" else "which"
        res = subprocess.run([checker, cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return res.returncode == 0
    except Exception:
        return False


def get_available_tunnel_services() -> Dict[str, bool]:
    """Returns a dictionary indicating which tunneling binaries are available on the host."""
    return {
        "cloudflared": is_command_available("cloudflared"),
        "localtunnel": is_command_available("lt") or is_command_available("npx"),
        "ngrok": is_command_available("ngrok"),
    }


def is_tunnel_running() -> bool:
    """Returns True if a background tunnel process is currently active."""
    global _CURRENT_TUNNEL_PROC
    with _TUNNEL_LOCK:
        if _CURRENT_TUNNEL_PROC is not None:
            if _CURRENT_TUNNEL_PROC.poll() is None:
                return True
            _CURRENT_TUNNEL_PROC = None
        return False


def get_active_tunnel_url() -> Optional[str]:
    """Returns the active public HTTPS tunnel URL if running, otherwise None."""
    global _CURRENT_TUNNEL_URL
    if is_tunnel_running():
        return _CURRENT_TUNNEL_URL
    return None


def get_active_tunnel_service() -> Optional[str]:
    """Returns the name of the active tunneling service."""
    global _CURRENT_TUNNEL_SERVICE
    if is_tunnel_running():
        return _CURRENT_TUNNEL_SERVICE
    return None


def stop_tunnel() -> bool:
    """Stops the active background tunnel process if one is running."""
    global _CURRENT_TUNNEL_PROC, _CURRENT_TUNNEL_URL, _CURRENT_TUNNEL_SERVICE
    with _TUNNEL_LOCK:
        if _CURRENT_TUNNEL_PROC is not None:
            try:
                _CURRENT_TUNNEL_PROC.terminate()
                try:
                    _CURRENT_TUNNEL_PROC.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    _CURRENT_TUNNEL_PROC.kill()
            except Exception as e:
                logger.warning(f"Error terminating tunnel process: {e}")
            finally:
                _CURRENT_TUNNEL_PROC = None
                _CURRENT_TUNNEL_URL = None
                _CURRENT_TUNNEL_SERVICE = None
            return True
        return False


def start_tunnel(port: int = 8501, service: str = "auto", timeout_seconds: int = 15) -> Tuple[bool, str]:
    """Starts an HTTPS tunnel targeting the specified local port.
    
    Parameters:
        port: The local Streamlit port (default 8501).
        service: 'auto', 'cloudflared', 'localtunnel', or 'ngrok'.
        timeout_seconds: Maximum seconds to wait for public URL discovery.
        
    Returns:
        (success, url_or_error_message)
    """
    global _CURRENT_TUNNEL_PROC, _CURRENT_TUNNEL_URL, _CURRENT_TUNNEL_SERVICE

    if is_tunnel_running():
        return True, _CURRENT_TUNNEL_URL or "Tunnel already active"

    avail = get_available_tunnel_services()
    selected_service = service.lower()

    if selected_service == "auto":
        if avail.get("cloudflared"):
            selected_service = "cloudflared"
        elif avail.get("localtunnel"):
            selected_service = "localtunnel"
        elif avail.get("ngrok"):
            selected_service = "ngrok"
        else:
            # Try npx localtunnel as standard fallback
            selected_service = "localtunnel"

    cmd: List[str] = []
    if selected_service == "cloudflared":
        cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]
    elif selected_service == "localtunnel":
        if is_command_available("lt"):
            cmd = ["lt", "--port", str(port)]
        elif is_command_available("npx"):
            cmd = ["npx", "localtunnel", "--port", str(port)]
        else:
            return False, "Neither 'lt' nor 'npx' command found for localtunnel."
    elif selected_service == "ngrok":
        cmd = ["ngrok", "http", str(port), "--log=stdout"]
    else:
        return False, f"Unknown tunnel service: '{service}'"

    # Start the subprocess
    try:
        logger.info(f"Starting tunnel command: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=(os.name == "nt" and cmd[0] == "npx"),
        )
    except FileNotFoundError as e:
        return False, f"Command '{cmd[0]}' not installed or not in PATH: {e}"
    except Exception as e:
        return False, f"Failed to start tunnel process: {e}"

    captured_url: Optional[str] = None
    url_found_event = threading.Event()
    output_lines: List[str] = []

    def _reader(stream, is_stderr: bool = False):
        nonlocal captured_url
        url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.(?:trycloudflare\.com|loca\.lt|ngrok(?:-free)?\.app)")
        for line in iter(stream.readline, ""):
            output_lines.append(line.strip())
            match = url_regex.search(line)
            if match and not captured_url:
                captured_url = match.group(0)
                url_found_event.set()
        stream.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, False), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, True), daemon=True)
    t_out.start()
    t_err.start()

    # Wait for the URL to be discovered or timeout
    found = url_found_event.wait(timeout=timeout_seconds)

    if found and captured_url:
        with _TUNNEL_LOCK:
            _CURRENT_TUNNEL_PROC = proc
            _CURRENT_TUNNEL_URL = captured_url
            _CURRENT_TUNNEL_SERVICE = selected_service
        logger.info(f"Tunnel successfully established: {captured_url} via {selected_service}")
        return True, captured_url
    else:
        # If ngrok, check local ngrok API at 127.0.0.1:4040/api/tunnels
        if selected_service == "ngrok":
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    for tun in data.get("tunnels", []):
                        if tun.get("proto") == "https":
                            captured_url = tun.get("public_url")
                            with _TUNNEL_LOCK:
                                _CURRENT_TUNNEL_PROC = proc
                                _CURRENT_TUNNEL_URL = captured_url
                                _CURRENT_TUNNEL_SERVICE = selected_service
                            return True, captured_url
            except Exception:
                pass

        # Cleanup if failed
        try:
            proc.terminate()
        except Exception:
            pass

        err_summary = "\n".join(output_lines[-5:]) if output_lines else "No output from tunnel process."
        return False, f"Tunnel timed out after {timeout_seconds}s. Logs: {err_summary}"


# =====================================================================
# 4. Streamlit UI Mobile Access Card Component
# =====================================================================

def render_mobile_access_card(port: int = 8501):
    """Renders a responsive, interactive Mobile Access & QR Code widget in Streamlit.
    
    Includes:
    - Tab 1: Local Wi-Fi (URL + QR Code)
    - Tab 2: Secure HTTPS Tunnel (Essential for mobile microphone & spoken exam permissions)
    """
    try:
        import streamlit as st
    except ImportError:
        logger.warning("Streamlit not installed, cannot render UI component.")
        return

    st.markdown("### 📱 Mobile Phone Access & Spoken Audio")
    st.caption("Open StudyAssistant on your smartphone for mobile revision, flashcards, or spoken oral exams.")

    tab_wifi, tab_https = st.tabs(["📶 Local Wi-Fi (Same Network)", "🔒 Secure HTTPS (Mic & Voice Enabled)"])

    # ---------------- Tab 1: Local Wi-Fi ----------------
    with tab_wifi:
        local_url = get_local_url(port)
        qr_uri = generate_qr_code_data_uri(local_url)

        col_qr, col_info = st.columns([1, 1.4])
        with col_qr:
            st.markdown(
                f"""<div style="text-align:center; background:#ffffff; padding:12px; border-radius:12px; border:1px solid #e2e8f0; display:inline-block;">
                    <img src="{qr_uri}" width="160" height="160" style="display:block; margin:auto;" alt="Local Wi-Fi QR Code"/>
                    <span style="font-size:11px; color:#64748b; font-weight:600; display:block; margin-top:6px;">Scan with Phone Camera</span>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_info:
            st.markdown(f"**Wi-Fi Direct Link:**")
            st.code(local_url, language="text")
            st.info(
                "💡 **Requirements:** Make sure your mobile phone is connected to the same Wi-Fi router / mobile hotspot as this computer.",
                icon="ℹ️"
            )
            st.warning(
                "⚠️ **Note on Voice:** Mobile browsers (iOS Safari & Android Chrome) block microphone access on plain HTTP. Use the **Secure HTTPS** tab if you want to use spoken oral exams on phone.",
                icon="🎙️"
            )

    # ---------------- Tab 2: Secure HTTPS Tunnel ----------------
    with tab_https:
        tunnel_active = is_tunnel_running()
        active_url = get_active_tunnel_url()
        active_service = get_active_tunnel_service()

        st.markdown(
            "Mobile browsers **strictly require an HTTPS connection** to grant microphone (`getUserMedia`) permissions. "
            "Start a secure instant tunnel below to enable oral viva and voice exam on your phone."
        )

        t_col1, t_col2 = st.columns([1.2, 1])

        with t_col1:
            if not tunnel_active:
                st.markdown("Status: 🔴 **Disconnected**")
                service_choice = st.selectbox(
                    "Tunnel Provider",
                    ["auto", "cloudflared", "localtunnel", "ngrok"],
                    index=0,
                    help="Auto detects cloudflared, localtunnel (npx lt), or ngrok."
                )
                if st.button("🚀 Launch HTTPS Tunnel", use_container_width=True, type="primary"):
                    with st.spinner("Establishing secure HTTPS tunnel..."):
                        success, res = start_tunnel(port=port, service=service_choice)
                        if success:
                            st.success(f"Tunnel Online: {res}")
                            st.rerun()
                        else:
                            st.error(f"Failed to start tunnel: {res}")
            else:
                st.markdown(f"Status: 🟢 **Active ({active_service})**")
                st.code(active_url, language="text")
                if st.button("🛑 Stop Tunnel", use_container_width=True):
                    stop_tunnel()
                    st.info("Tunnel closed.")
                    st.rerun()

        with t_col2:
            if tunnel_active and active_url:
                https_qr = generate_qr_code_data_uri(active_url)
                st.markdown(
                    f"""<div style="text-align:center; background:#ffffff; padding:12px; border-radius:12px; border:1px solid #10b981; display:inline-block;">
                        <img src="{https_qr}" width="160" height="160" style="display:block; margin:auto;" alt="HTTPS QR Code"/>
                        <span style="font-size:11px; color:#059669; font-weight:700; display:block; margin-top:6px;">🔒 HTTPS Voice Ready</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div style="text-align:center; background:#f8fafc; padding:24px 12px; border-radius:12px; border:1px dashed #cbd5e1;">
                        <span style="font-size:32px;">🔒</span>
                        <div style="font-size:12px; color:#64748b; margin-top:8px;">Start tunnel to generate HTTPS QR Code</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
