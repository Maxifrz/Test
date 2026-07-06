/* ECKERT Website — schlankes Vanilla-JS (ersetzt die Prototyp-Runtime).
   Funktionen: Kontaktformular-Validierung + Versand an das Backend,
   Karte-auf-Klick (Datenschutz), rel-Härtung externer Links.
   Keine Cookies, kein Tracking. */
"use strict";
(function () {
  // Versand-Endpunkt (Plattform-Backend, siehe nginx-Website-vHost).
  // Überschreibbar, falls die Website getrennt vom Tool gehostet wird:
  //   <script>window.EK_CONTACT_ENDPOINT = "https://tool.example/api/public/contact";</script>
  var CONTACT_ENDPOINT = window.EK_CONTACT_ENDPOINT || "/api/public/contact";

  // ---------- Kontaktformular ----------
  var form = document.querySelector("form");
  var nameEl = document.getElementById("ek-name");
  if (form && nameEl) {
    var emailEl = document.getElementById("ek-email");
    var phoneEl = document.getElementById("ek-phone");
    var standortEl = document.getElementById("ek-standort");
    var rolleEl = document.getElementById("ek-rolle");
    var anliegenEl = document.getElementById("ek-anliegen");
    var consentEl = form.querySelector('input[type="checkbox"]');

    // Honeypot: unsichtbares Feld, das nur Bots ausfüllen (Backend verwirft dann)
    var honeypot = document.createElement("input");
    honeypot.type = "text";
    honeypot.name = "firma";
    honeypot.tabIndex = -1;
    honeypot.autocomplete = "off";
    honeypot.setAttribute("aria-hidden", "true");
    honeypot.style.cssText = "position:absolute;left:-9999px;width:1px;height:1px;opacity:0;";
    form.appendChild(honeypot);

    function setErr(el, msg) {
      if (!el) return !msg;
      el.setAttribute("aria-invalid", msg ? "true" : "false");
      el.style.borderColor = msg ? "var(--bad)" : "";
      var id = el.id + "-err";
      var node = document.getElementById(id);
      if (msg) {
        if (!node) {
          node = document.createElement("p");
          node.id = id;
          node.setAttribute("role", "alert");
          node.style.cssText = "margin:6px 0 0;font-size:12.5px;color:var(--bad);";
          el.insertAdjacentElement("afterend", node);
        }
        node.textContent = msg;
      } else if (node) {
        node.remove();
      }
      return !msg;
    }

    form.setAttribute("novalidate", "true");
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var ok = true;
      ok = setErr(nameEl, nameEl.value.trim() ? "" : "Bitte geben Sie Ihren Namen an.") && ok;
      ok = setErr(emailEl, /.+@.+\..+/.test(emailEl && emailEl.value || "") ? "" : "Bitte geben Sie eine gültige E-Mail-Adresse an.") && ok;
      ok = setErr(anliegenEl, anliegenEl && anliegenEl.value.trim().length >= 10 ? "" : "Bitte beschreiben Sie Ihr Anliegen (mind. 10 Zeichen).") && ok;
      if (consentEl && !consentEl.checked) {
        consentEl.style.outline = "2px solid var(--bad)";
        consentEl.style.outlineOffset = "2px";
        ok = false;
      } else if (consentEl) {
        consentEl.style.outline = "";
      }
      if (!ok) return;

      var submitBtn = form.querySelector('button[type="submit"], button:not([type])');
      if (submitBtn) { submitBtn.disabled = true; submitBtn.style.opacity = "0.6"; }
      var errNode = document.getElementById("ek-form-err");
      if (errNode) errNode.remove();

      function showSuccess() {
        var card = document.createElement("div");
        card.setAttribute("role", "status");
        card.style.cssText = "background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ok);border-radius:2px;padding:28px 30px;";
        card.innerHTML =
          '<p style="font-family:var(--font-serif);font-size:22px;margin:0 0 8px;">Vielen Dank — Ihre Nachricht ist angekommen.</p>' +
          '<p style="font-size:14.5px;line-height:1.6;color:var(--ink-2);margin:0;">Wir melden uns innerhalb von 24 Stunden. In dringenden Fällen erreichen Sie die Krisen-Hotline unter ⟨ Telefonnummer ⟩.</p>';
        form.replaceWith(card);
        card.scrollIntoView({ block: "center", behavior: "smooth" });
      }

      function showError() {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.style.opacity = ""; }
        var node = document.createElement("div");
        node.id = "ek-form-err";
        node.setAttribute("role", "alert");
        node.style.cssText = "background:var(--card);border:1px solid var(--line);border-left:3px solid var(--bad);border-radius:2px;padding:16px 18px;margin:0 0 16px;font-size:14px;line-height:1.6;color:var(--ink-2);";
        node.textContent = "Die Nachricht konnte gerade nicht übermittelt werden. Bitte versuchen Sie es erneut oder rufen Sie uns an: ⟨ Telefonnummer ⟩.";
        form.insertBefore(node, form.firstChild);
        node.scrollIntoView({ block: "center", behavior: "smooth" });
      }

      var standortVal = standortEl && standortEl.value !== "Standort wählen" ? standortEl.value : null;
      var rolleVal = (rolleEl && rolleEl.value) || null;

      fetch(CONTACT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: nameEl.value.trim(),
          email: emailEl.value.trim(),
          phone: (phoneEl && phoneEl.value.trim()) || null,
          standort: standortVal,
          rolle: rolleVal,
          message: anliegenEl.value.trim(),
          consent: true,
          firma: honeypot.value || null
        })
      })
        .then(function (res) {
          if (res.ok) showSuccess(); else showError();
        })
        .catch(showError);
    });
  }

  // ---------- Karte erst auf Klick laden (Datenschutz) ----------
  document.querySelectorAll("button, a").forEach(function (el) {
    if (el.textContent.trim().indexOf("Karte laden") !== 0) return;
    el.addEventListener("click", function (e) {
      e.preventDefault();
      var box = el.closest("div");
      var iframe = document.createElement("iframe");
      iframe.title = "Karte";
      iframe.loading = "lazy";
      iframe.style.cssText = "width:100%;height:100%;min-height:320px;border:0;display:block;";
      // Platzhalter-Koordinaten (Hannover Mitte) — vor Go-Live je Standort setzen
      iframe.src = "https://www.openstreetmap.org/export/embed.html?bbox=9.70%2C52.35%2C9.78%2C52.40&layer=mapnik";
      if (box) { box.innerHTML = ""; box.appendChild(iframe); }
    });
  });

  // ---------- Mobile Navigation (Burger, ≤920px via responsive.css) ----------
  var header = document.querySelector("header");
  var nav = header && header.querySelector('nav[aria-label="Hauptnavigation"]');
  if (header && nav) {
    var bar = nav.parentElement;
    var burger = document.createElement("button");
    burger.className = "ek-burger";
    burger.setAttribute("aria-label", "Menü öffnen");
    burger.setAttribute("aria-expanded", "false");
    burger.innerHTML =
      '<svg width="20" height="20" viewBox="0 0 20 20" stroke="currentColor" stroke-width="1.6" fill="none">' +
      '<path d="M3 5h14M3 10h14M3 15h14"/></svg>';

    var menu = document.createElement("nav");
    menu.className = "ek-mobilemenu";
    menu.setAttribute("aria-label", "Mobile Navigation");
    nav.querySelectorAll("a").forEach(function (a) {
      menu.appendChild(a.cloneNode(true));
    });
    // Krisen-Hotline-CTA (falls vorhanden) ans Menüende
    var cta = bar.querySelector('a[href*="kontakt"]');
    if (cta && /Hotline/i.test(cta.textContent)) {
      var m = cta.cloneNode(true);
      m.removeAttribute("style");
      m.style.cssText = "display:block;padding:13px 4px;font-weight:600;color:var(--accent);text-decoration:none;border-bottom:none;";
      menu.appendChild(m);
    }

    bar.appendChild(burger);
    header.style.position = "sticky";
    header.appendChild(menu);

    burger.addEventListener("click", function () {
      var open = menu.classList.toggle("open");
      burger.setAttribute("aria-expanded", open ? "true" : "false");
      burger.setAttribute("aria-label", open ? "Menü schließen" : "Menü öffnen");
    });
    menu.addEventListener("click", function (e) {
      if (e.target.closest("a")) menu.classList.remove("open");
    });
  }

  // ---------- Externe Links härten ----------
  document.querySelectorAll('a[href^="http"]').forEach(function (a) {
    if (a.hostname !== location.hostname) a.rel = "noopener noreferrer";
  });
})();
