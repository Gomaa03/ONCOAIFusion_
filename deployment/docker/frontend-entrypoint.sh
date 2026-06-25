#!/bin/sh

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   🧬 OncoAI Fusion - Frontend Ready!                         ║"
echo "║                                                              ║"
echo "║   🌐 Open in browser: http://localhost:3000                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

exec nginx -g 'daemon off;'
