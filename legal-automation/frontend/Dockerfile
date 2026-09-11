# --- Build ---
FROM node:22-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- Runtime ---
# nginx statt `npm install -g serve`: `serve` ist ein Entwicklungswerkzeug,
# wurde ungepinnt bei jedem Build neu aufgeloest (nicht reproduzierbar) und
# liefert weder Kompression noch Cache-Header. nginx bringt beides mit und
# laeuft als non-root.
FROM nginx:1.27-alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx-spa.conf /etc/nginx/conf.d/default.conf

# Der offizielle Alpine-Image laeuft als root, um Port 80 zu binden. Wir
# hoeren auf 3000 (unprivilegiert) und koennen deshalb den nginx-User nutzen.
RUN chown -R nginx:nginx /usr/share/nginx/html /var/cache/nginx \
    && touch /var/run/nginx.pid && chown nginx:nginx /var/run/nginx.pid
USER nginx

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD wget -q --spider http://127.0.0.1:3000/ || exit 1
