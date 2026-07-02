"use strict";
(function () {
  var token = new URLSearchParams(window.location.search).get("token");
  var base = "/api/public/creditor-claims/";

  var globalError = document.getElementById("globalError");
  var content = document.getElementById("content");
  var matterInfo = document.getElementById("matterInfo");
  var form = document.getElementById("claimForm");
  var formCard = document.getElementById("formCard");
  var successCard = document.getElementById("successCard");
  var successDetail = document.getElementById("successDetail");
  var submitBtn = document.getElementById("submitBtn");

  function showError(msg) {
    globalError.textContent = msg;
    globalError.classList.remove("hidden");
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  if (!token) {
    showError("Kein Zugangstoken gefunden. Bitte verwenden Sie den vollständigen Link aus Ihrem Schreiben der Kanzlei.");
    return;
  }

  // 1) Token prüfen + Verfahren anzeigen
  fetch(base + encodeURIComponent(token))
    .then(function (res) {
      if (!res.ok) throw new Error(res.status === 404 ? "ungültig" : "fehler");
      return res.json();
    })
    .then(function (data) {
      matterInfo.innerHTML =
        "<strong>Insolvenzverfahren</strong> &nbsp;·&nbsp; <strong>Aktenzeichen:</strong> " +
        esc(data.matter_number);
      content.classList.remove("hidden");
    })
    .catch(function (err) {
      if (err.message === "ungültig") {
        showError("Dieser Anmelde-Link ist ungültig oder wurde deaktiviert. Bitte wenden Sie sich an die Kanzlei.");
      } else {
        showError("Das Portal ist derzeit nicht erreichbar. Bitte versuchen Sie es später erneut.");
      }
    });

  // 2) Forderung anmelden
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    submitBtn.disabled = true;

    var payload = {
      creditor_name: document.getElementById("creditor_name").value.trim(),
      claim_amount: document.getElementById("claim_amount").value,
      creditor_email: document.getElementById("creditor_email").value.trim() || null,
      creditor_address: document.getElementById("creditor_address").value.trim() || null,
      creditor_reference: document.getElementById("creditor_reference").value.trim() || null,
      claim_reason: document.getElementById("claim_reason").value.trim() || null,
    };

    fetch(base + encodeURIComponent(token), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.detail || "Anmeldung fehlgeschlagen");
          return body;
        });
      })
      .then(function (body) {
        successDetail.textContent =
          "Ihre Forderung wurde unter der lfd. Nr. " + body.claim_number +
          " erfasst. " + (body.hinweis || "");
        formCard.classList.add("hidden");
        successCard.classList.remove("hidden");
        window.scrollTo(0, 0);
      })
      .catch(function (err) {
        showError(err.message);
        submitBtn.disabled = false;
      });
  });
})();
