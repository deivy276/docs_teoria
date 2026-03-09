# WP4 — Sector primordial de la TUO

## Estado del paquete
**Congelado como formulación primordial efectiva de referencia**.

## Objetivo
Cerrar el **sector primordial** de la TUO al nivel necesario para una EFT cosmológica seria, sin fingir todavía una teoría inflacionaria/fundamental completa del tejido.

La filosofía de WP4 es:

> **Separar con honestidad lo que la TUO ya puede asumir de forma consistente sobre el universo primordial de lo que aún no puede derivar desde primeros principios.**

---

## 1. Decisión principal de WP4

La TUO adopta, por ahora, una **clausura primordial conservadora**:

1. las perturbaciones primordiales relevantes son **adiabáticas**,
2. son **casi gaussianas**,
3. tienen espectro **casi invariante de escala**,
4. y se parametrizan por los observables estándar
   \(A_s\), \(n_s\) (y opcionalmente \(r\), \(\alpha_s\)).

Esto implica que, en la versión actual de la TUO:

- **la generación primordial no se atribuye todavía de forma única al tejido**,
- pero **la propagación cosmológica posterior sí está gobernada por la TUO**.

En otras palabras:

> **WP4 congela la TUO como una teoría efectiva de evolución cosmológica posterior a la generación primordial, con una semilla inicial estándar y físicamente consistente.**

---

## 2. Hipótesis primordial de referencia

### 2.1 Modo escalar primordial

El espectro primordial de perturbaciones de curvatura se fija como

\[
\mathcal P_{\mathcal R}(k)
=
A_s\left(\frac{k}{k_*}\right)^{n_s-1 + \frac12\alpha_s\ln(k/k_*)}
\]

con pivote cosmológico de referencia:

\[
k_* = 0.05\;\mathrm{Mpc}^{-1}.
\]

### 2.2 Rama de referencia actual

En la EFT actual de la TUO se congela por defecto:

\[
\alpha_s = 0,
\qquad
r = 0,
\qquad
f_{\rm NL}=0,
\]

salvo que un módulo observacional posterior requiera liberar alguno de estos parámetros.

Esto define una **rama primordial mínima** y operacional.

---

## 3. Relación con el campo del tejido

El campo efectivo del tejido \(\phi\) satisface, en cosmología homogénea,

\[
\ddot \phi + 3H\dot\phi + V_{,\phi} = \alpha(\phi)\rho_c,
\qquad
\alpha(\phi)=\frac{d\beta(\phi)}{d\phi}.
\]

En la era dominada por radiación:

\[
T_r = -\rho_r + 3P_r = 0,
\]

de modo que el campo del tejido no es fuenteado directamente por la radiación.

### Consecuencia primordial importante
Durante la fase radiativa temprana, y más aún si

\[
3H\dot\phi \gg V_{,\phi},
\]

el campo queda aproximadamente congelado:

\[
\dot\phi \approx 0.
\]

Esto implica que, en la formulación primordial actual:

- la TUO **no genera por sí sola** una fuente fuerte de isocurvatura en el sector radiativo,
- y el acoplamiento oscuro primordial queda suprimido porque

\[
Q \propto \beta_0\dot\phi\rho_c \approx 0.
\]

Esta es la justificación covariante del **bloqueo radiativo primordial**.

---

## 4. Condiciones iniciales de perturbaciones

### 4.1 Sector adiabático

La rama de referencia de la TUO adopta:

\[
\mathcal S_{bc}=0,
\qquad
\mathcal S_{r\phi}=0,
\qquad
\mathcal S_{c\phi}=0
\]

al inicio del régimen cosmológico efectivo.

Es decir:

> **No se introducen modos isocurvatura primordiales en la versión de referencia de la TUO.**

### 4.2 Campo del tejido en perturbaciones

En la formulación actual, las perturbaciones primordiales del tejido se congelan en la rama mínima de referencia de forma que el sector observable queda dominado por el modo de curvatura primordial \(\mathcal R\).

Eso puede implementarse, según el gauge, como una de estas dos clausuras equivalentes en IR:

- \(\delta\phi_{\rm ini}=0\) en gauge apropiado,
- o bien \(\delta\phi\) completamente alineado con el modo adiabático, sin isocurvatura independiente.

WP4 congela la siguiente convención de trabajo:

\[
\boxed{\text{La TUO actual se inicia en la rama adiabática pura, sin isocurvatura primordial independiente del tejido.}}
\]

---

## 5. Normalización primordial del crecimiento

### 5.1 Principio

El crecimiento lineal de la TUO debe quedar anclado a la **misma amplitud primordial** que usa el modelo estándar, no a un \(\sigma_8\) libre ajustado ad hoc.

Por ello, WP4 congela como principio:

\[
A_s \rightarrow \sigma_{8,0}^{\rm pred}
\]

mediante una construcción de referencia consistente.

### 5.2 Regla operativa actual (EFT)

Mientras no exista un solver Einstein–Boltzmann completo de la TUO, la normalización se define por una regla híbrida:

1. se toma una referencia \(\Lambda\)CDM con los mismos
   \((h,\omega_b,\omega_c,A_s,n_s)\),
2. se calcula \(\sigma_{8,0}^{\rm ref}\),
3. se usa el crecimiento relativo de la TUO para transportar esa amplitud a \(z=0\).

Esquemáticamente:

\[
\sigma_{8,0}^{\rm TUO}
=
\sigma_{8,0}^{\rm ref}(h,\omega_b,\omega_c,A_s,n_s)
\times
\frac{D_{\rm TUO}(z_{\rm match}\to 0)}{D_{\Lambda\rm CDM}(z_{\rm match}\to 0)}.
\]

Esto no pretende ser aún la derivación primordial definitiva, pero sí una **normalización físicamente honesta**.

### 5.3 Criterio de consistencia

En el límite \(\beta_0\to 0\), \(\phi\to\phi_*\), la TUO debe reproducir:

\[
\sigma_{8,0}^{\rm TUO} \to \sigma_{8,0}^{\Lambda\rm CDM}.
\]

Ese criterio ya forma parte obligatoria de la validación de WP4.

---

## 6. Qué afirma y qué no afirma el sector primordial actual

### Sí afirma
- existe una semilla primordial estándar físicamente consistente;
- la TUO no destruye la física temprana;
- el bloqueo radiativo suprime el acoplamiento oscuro temprano;
- el crecimiento tardío puede derivarse de una amplitud primordial \(A_s\).

### No afirma todavía
- que la TUO ya tenga una teoría inflacionaria propia cerrada;
- que \(A_s\), \(n_s\), \(r\) se deriven desde el tejido;
- que el modelo prediga ya no-gaussianidades o tensores primordiales propios;
- que la TUO reemplace completamente al sector primordial estándar.

WP4 deja esto explícito para evitar sobreinterpretaciones.

---

## 7. Límite \(\Lambda\)CDM en el sector primordial

La TUO primordial actual contiene a \(\Lambda\)CDM como límite, en el sentido de que:

- la semilla primordial adiabática estándar se recupera,
- la evolución del tejido se congela,
- y la amplitud lineal resultante reproduce el comportamiento estándar.

Por tanto:

\[
\boxed{\Lambda\mathrm{CDM} \subset \mathrm{TUO}\ \text{también en la clausura primordial efectiva actual.}}
\]

---

## 8. Qué falta para cerrar el sector primordial fundamental

WP4 deja definidos tres niveles:

### Nivel A — ya cerrado
Sector primordial **efectivo** con semilla adiabática estándar y crecimiento anclado a \(A_s\).

### Nivel B — pendiente a medio plazo
Acoplar la TUO a un solver tipo Einstein–Boltzmann completo para que \(A_s\to\sigma_8\) deje de depender de una normalización híbrida.

### Nivel C — completion fundamental
Desarrollar una teoría primordial propia del tejido que produzca:

- \(A_s\),
- \(n_s\),
- \(r\),
- y quizá no-gaussianidades,

como predicciones del sector del tejido.

---

## 9. Declaración oficial de WP4

La formulación primordial de referencia de la TUO queda congelada así:

\[
\boxed{
\mathcal P_{\mathcal R}(k)=A_s\left(\frac{k}{k_*}\right)^{n_s-1},
\qquad
\text{rama adiabática pura,}
\qquad
Q_{\rm early}\approx 0,
\qquad
A_s\to\sigma_8\ \text{por normalización híbrida consistente.}
}
\]

En palabras:

> **La TUO actual adopta una clausura primordial conservadora: semilla adiabática estándar, bloqueo radiativo natural del tejido y crecimiento tardío gobernado por la dinámica efectiva del campo \(\phi\).**

---

## 10. Resultado conceptual de WP4

WP4 no pretende afirmar que la TUO ya tenga una inflación propia, sino que:

> **la TUO ya dispone de un sector primordial efectivo honesto, suficiente para conectar una semilla primordial estándar con las predicciones cosmológicas lineales observables del modelo.**

Ese es el cierre primordial mínimo necesario para una EFT cosmológica seria.
