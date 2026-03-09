
# WP2 — Prueba de estabilidad y ausencia de fantasmas de la TUO

## Estado
**WP2 queda definido y parcialmente cerrado a nivel de la EFT cosmológica actual.**

- **Sí queda establecido** que la **EFT escalar–tensorial actual** de la TUO es una teoría de segundo orden, con un grado de libertad escalar extra canónico y, por tanto, **sin fantasma de Ostrogradski en el régimen IR cosmológico**.
- **No queda todavía demostrado** que una posible **completion tensorial UV** basada en \(A_{\mu\nu}\) sea ghost-free. Eso permanece como una tarea abierta.

---

## 1. Objetivo de WP2

El objetivo de WP2 es separar de forma rigurosa dos niveles de estabilidad:

1. **Estabilidad de la EFT cosmológica actual**  
   Probar que la teoría efectiva utilizada en background y crecimiento lineal no introduce grados de libertad patológicos.

2. **Estabilidad de una completion tensorial subyacente**  
   Formular el problema correcto para demostrar (o refutar) que el origen tensorial de la TUO puede cerrarse sin fantasmas.

---

## 2. Acción efectiva de referencia

La EFT actual de la TUO se congela como

\[
S_{\rm TUO}^{\rm eff}
=
\int d^4x\,\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\nabla\phi)^2
-
V(\phi)
\right]
+
S_b[g_{\mu\nu},\psi_b]
+
S_r[g_{\mu\nu},\psi_r]
+
S_c[\tilde g^{(c)}_{\mu\nu},\psi_c],
\]

con

\[
\tilde g^{(c)}_{\mu\nu}=e^{2\beta(\phi)}g_{\mu\nu},
\qquad
\beta(\phi)\simeq \beta_0\,\phi/M_{\rm Pl},
\]

y potencial mínimo

\[
V(\phi)=V_0+\frac12\nu^2\phi^2+\frac{\lambda_4}{4}\phi^4.
\]

---

## 3. Resultado central de WP2 a nivel EFT

### Proposición P2.1 — ausencia de fantasma en la EFT actual

Bajo la acción anterior, el sector \((g_{\mu\nu},\phi)\) de la TUO:

- tiene ecuaciones de movimiento de **segundo orden**;
- propaga en el régimen IR cosmológico:
  \[
  2_{\rm tensoriales}+1_{\rm escalar};
  \]
- no contiene un término cinético de signo incorrecto para \(\phi\);
- y, por tanto, **no presenta un ghost de Ostrogradski** en su formulación efectiva actual.

### Justificación

1. El lagrangiano escalar es canónico:
   \[
   \mathcal L_\phi = -\frac12(\nabla\phi)^2 - V(\phi).
   \]
   El coeficiente cinético es positivo.

2. La ecuación del campo
   \[
   \Box\phi - V_{,\phi} = \alpha(\phi)T_c
   \]
   es de segundo orden.

3. El acoplamiento conforme al sector oscuro
   \[
   \tilde g^{(c)}_{\mu\nu}=e^{2\beta(\phi)}g_{\mu\nu}
   \]
   depende de \(\phi\), pero **no de derivadas de \(\phi\)**. Por tanto, no introduce derivadas superiores ocultas en las ecuaciones de la materia oscura.

4. El sector tensorial gravitacional sigue siendo el de GR en el marco efectivo actual.

### Conclusión
A nivel de EFT cosmológica, la TUO **sí queda libre de fantasmas**.

---

## 4. Condiciones de estabilidad que deben imponerse explícitamente

Para que la formulación efectiva sea físicamente aceptable, la TUO debe respetar estas condiciones:

### C1. Masa de Planck efectiva positiva
\[
M_{\rm Pl}^2 > 0.
\]

### C2. Coeficiente cinético positivo del escalar
\[
Z(\phi)>0.
\]
En la versión mínima actual:
\[
Z(\phi)=1.
\]

### C3. Velocidad de propagación escalar real y positiva
En la versión canónica:
\[
c_s^2 = 1.
\]
No debe aparecer:
\[
c_s^2 < 0.
\]

### C4. Estabilidad tensorial
El sector tensorial debe conservar:
\[
Q_T>0,\qquad c_T^2>0.
\]
En la versión efectiva actual:
\[
c_T^2=1.
\]

### C5. Gravedad efectiva positiva en crecimiento
La corrección fenomenológica de crecimiento debe satisfacer

\[
\mu(k,a)=1+\frac{2\beta_0^2 k^2}{k^2+a^2m_{\rm eff}^2(a)} > 0.
\]

### C6. Masa efectiva del escalar no patológicamente taquiónica
\[
m_{\rm eff}^2 = V_{,\phi\phi}
\]
puede ser pequeña o incluso levemente negativa en regímenes transitorios, pero no debe inducir crecimiento explosivo en escalas cosmológicas incompatibles con observación.

---

## 5. Bloqueo radiativo y estabilidad

El “bloqueo radiativo” no es un parche y tampoco introduce inestabilidad.

Como
\[
T = -\rho + 3P,
\]
durante dominación radiativa:
\[
P_r=\rho_r/3 \quad \Longrightarrow \quad T_r=0.
\]

Entonces la radiación no fuentea directamente al campo \(\phi\), lo que favorece:

\[
\dot\phi \approx 0
\]

por fricción de Hubble en el universo temprano.

**Resultado:** el mecanismo de congelamiento temprano del tejido es compatible con estabilidad covariante.

---

## 6. Qué NO queda demostrado todavía

La ausencia de fantasmas **no está demostrada** todavía para una possible completion tensorial UV del tipo

\[
A_{\mu\nu}=\frac14\phi\,g_{\mu\nu}+\hat A_{\mu\nu}.
\]

En particular, siguen abiertos:

1. el conteo exacto de grados de libertad de \(\hat A_{\mu\nu}\);
2. la posibilidad de modos vectoriales o escalares extra no deseados;
3. la existencia de un ghost en una completion tensorial no lineal;
4. la necesidad de una simetría más fuerte (TDiff/WTDiff, Fierz–Pauli, etc.).

---

## 7. Programa formal restante para cerrar WP2 completamente

### WP2.A — Prueba EFT (ya alcanzada)
Probar que la EFT actual es de segundo orden y libre de ghosts.
**Estado:** alcanzado.

### WP2.B — Análisis de perturbaciones cuadráticas
Derivar la acción cuadrática para perturbaciones cosmológicas y verificar explícitamente:
- \(Q_s>0\),
- \(c_s^2>0\),
- \(Q_T>0\),
- \(c_T^2>0\).

**Estado:** parcialmente implícito; pendiente de formalización escrita completa.

### WP2.C — Conteo Hamiltoniano de la completion tensorial
Realizar una formulación ADM/Hamiltoniana del sector padre \(A_{\mu\nu}\) para decidir si la reducción a \(\phi\) puede hacerse sin fantasmas.

**Estado:** abierto.

---

## 8. Dictamen oficial de WP2

### Resultado principal
**La TUO actual está libre de fantasmas a nivel de EFT cosmológica.**

### Resultado pendiente
**La completion tensorial UV de la TUO no está todavía demostrada como ghost-free.**

---

## 9. Frase de cierre recomendada

> La prueba de estabilidad de la TUO se considera cerrada a nivel de teoría efectiva cosmológica: el sistema \((g_{\mu\nu},\phi)\) define una EFT escalar–tensorial de segundo orden, con un único grado de libertad escalar adicional y sin indicios de fantasmas. Lo que permanece abierto no es la EFT cosmológica, sino su posible completion tensorial fundamental.

---

## 10. Entregables de WP2

- **D2.1**: este documento de cierre conceptual de estabilidad EFT.
- **D2.2** (pendiente): derivación escrita de la acción cuadrática de perturbaciones.
- **D2.3** (pendiente): análisis Hamiltoniano del sector tensorial padre.

---

## 11. Criterio de éxito de WP2

WP2 se considera **superado en su núcleo** si se acepta la TUO como EFT cosmológica.  
WP2 solo quedará **cerrado totalmente** cuando se resuelva o descarte de forma concluyente la completion tensorial UV sin fantasmas.
