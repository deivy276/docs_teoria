# WP8 — Estabilidad cuántica y naturalidad de la TUO

## Estado
**Congelado (plan de trabajo formal).**

## Propósito
Definir el marco mínimo con el que debe evaluarse la **estabilidad cuántica** y la **naturalidad** de la TUO, entendida en esta etapa como una **teoría efectiva de campos cosmológica** (EFT) válida por debajo de una escala de corte \(\Lambda_{\rm EFT}\).

La meta de WP8 no es todavía construir una UV completion completa, sino fijar:

- qué significa que la TUO sea *cuánticamente aceptable* como EFT,
- qué cantidades deben permanecer radiativamente estables,
- qué jerarquías requieren protección por simetría,
- y bajo qué condiciones la teoría puede considerarse técnicamente natural en el rango cosmológico de interés.

---

## Hipótesis de trabajo congeladas

### H8.1 — La TUO actual se interpreta como EFT infrarroja
La formulación cosmológica actual de la TUO se considera válida para energías y curvaturas por debajo de una escala de corte \(\Lambda_{\rm EFT}\), sin pretender todavía describir la gravitación cuántica completa ni el régimen UV profundo.

La acción de referencia es la congelada en WP0–WP4:

\[
S_{\rm TUO}^{\rm eff}
=
\int d^4x\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\partial\phi)^2
-V(\phi)
\right]
+
S_b[g]
+
S_r[g]
+
S_c[e^{2\beta(\phi)}g],
\]

con

\[
V(\phi)=V_0+\frac12\nu^2\phi^2+\frac{\lambda_4}{4}\phi^4,
\qquad
\beta(\phi)=\beta_0\phi/M_{\rm Pl}
\]

en la versión mínima.

### H8.2 — Distinción entre estabilidad clásica, radiativa y naturalidad
WP8 congela que deben distinguirse tres niveles:

1. **Estabilidad clásica**: ausencia de fantasmas, gradientes patológicos o taquiones catastróficos (cubierto en WP2 a nivel EFT).
2. **Estabilidad radiativa**: los parámetros del modelo no reciben correcciones cuánticas que destruyan la EFT en su dominio de validez.
3. **Naturalidad**: la pequeñez de ciertas escalas (en particular la masa del tejido y la energía de vacío efectiva) no depende de cancelaciones arbitrarias descontroladas.

### H8.3 — Protección simétrica aún no fijada de forma definitiva
WP8 congela que la TUO **todavía no dispone** de una simetría UV completamente establecida que proteja automáticamente todos sus parámetros. Por tanto, la naturalidad se tratará en dos niveles:

- **mínimo**: consistencia EFT y estabilidad radiativa aceptable en el dominio cosmológico;
- **fuerte**: protección por simetría (por ejemplo, simetría de desplazamiento aproximada, simetría conforme/escala, o mecanismo de sequestering), dejada como objetivo de desarrollo posterior.

### H8.4 — El problema de \(V_0\) se congela como problema abierto compartido
WP8 congela que la pequeña escala efectiva tipo energía oscura,

\[
V_0 \sim \rho_{\rm DE},
\]

se considera, en esta etapa, un problema abierto de naturalidad **compartido** con la mayoría de modelos de energía oscura y con \(\Lambda\)CDM mismo. La TUO no se declara resuelta en este punto.

---

## Objetivos científicos de WP8

### O8.1 — Evaluar estabilidad radiativa de los parámetros del tejido
Determinar si las correcciones de loop a:

- \(\nu^2\),
- \(\lambda_4\),
- \(\beta_0\),
- y \(Z(\phi)\)

son lo suficientemente pequeñas como para no invalidar la descripción cosmológica efectiva.

### O8.2 — Establecer una escala de corte EFT razonable
Determinar cuál es la ventana de validez de la TUO como EFT:

\[
H_0 \ll E \ll \Lambda_{\rm EFT}.
\]

La meta es fijar una escala de corte operativa consistente con el sector oscuro, el crecimiento lineal y la cosmología de fondo.

### O8.3 — Identificar qué parámetros requieren protección por simetría
Distinguir entre:

- parámetros técnicamente naturales,
- parámetros aceptables como libres de EFT,
- y parámetros que exigen una explicación más profunda.

### O8.4 — Preparar el puente hacia una completion UV
WP8 debe dejar claro qué estructura simétrica o UV completion sería deseable para cerrar la teoría más allá de la EFT.

---

## Marco EFT congelado

La acción wilsoniana mínima de referencia se congela como:

\[
S_{\rm EFT}
=
\int d^4x\sqrt{-g}
\Bigg[
\frac{M_{\rm Pl}^2}{2}R
-\frac12 Z(\phi)(\partial\phi)^2
-V(\phi)
+\sum_i \frac{c_i}{\Lambda_{\rm EFT}^{\Delta_i-4}}\,\mathcal O_i
\Bigg]
+
S_b[g]
+
S_r[g]
+
S_c[e^{2\beta(\phi)}g].
\]

Aquí:

- \(\mathcal O_i\) representan operadores irrelevantes compatibles con las simetrías de la EFT,
- \(\Delta_i\) es su dimensión,
- \(c_i\) son coeficientes adimensionales wilsonianos.

La política congelada de WP8 es:

> **La TUO se considera cuánticamente aceptable si los operadores adicionales están suprimidos por \(\Lambda_{\rm EFT}\) y no reordenan violentamente la jerarquía cosmológica en el rango de energías relevante.**

---

## Correcciones radiativas esquemáticas congeladas

A nivel de análisis dimensional (NDA), WP8 congela como estimaciones de trabajo:

### Masa efectiva del tejido
\[
\delta \nu^2 \sim \frac{c_m}{16\pi^2}\,\Lambda_{\rm EFT}^2
\]

para acoplamientos escalares genéricos no protegidos.

### Auto-interacción
\[
\delta \lambda_4 \sim \frac{c_\lambda}{16\pi^2}\,\lambda_4^2 + \cdots
\]

### Término de vacío
\[
\delta V_0 \sim \frac{c_0}{16\pi^2}\,\Lambda_{\rm EFT}^4
\]

### Acoplamiento oscuro
En la EFT mínima, el running de \(\beta_0\) se congela como pequeño en el rango cosmológico,

\[
\delta \beta_0 \sim \frac{c_\beta}{16\pi^2}\,\beta_0\ln\frac{\Lambda_{\rm EFT}}{\mu}
\]

a menos que una completion específica indique lo contrario.

---

## Criterios de aceptabilidad radiativa

WP8 congela los siguientes criterios mínimos.

### C1 — Masa del tejido
El modelo se considerará radiativamente aceptable si, en la ventana de validez elegida,

\[
|\delta \nu^2| \lesssim \nu^2
\]

o si existe una simetría claramente identificada que justifique la pequeñez observada.

### C2 — Acoplamiento oscuro
El acoplamiento conforme al sector oscuro será aceptable si el running radiativo no induce:

- acoplamientos bariónicos relevantes,
- ni una deriva de \(\beta_0\) incompatible con el análisis cosmológico.

### C3 — Término cinético
El renormalizado efectivo debe satisfacer:

\[
Z_{\rm eff}(\phi) > 0
\]

en el dominio de campo explorado cosmológicamente.

### C4 — Operadores superiores
Los operadores de dimensión alta deben permanecer subdominantes para los observables calculados en WP0–WP7.

---

## Naturalidad: política congelada

WP8 distingue entre dos nociones.

### Naturalidad débil (aceptable en EFT cosmológica)
La TUO se considerará aceptable si:

- sus parámetros libres son estables dentro del rango cosmológico relevante,
- la teoría no exige cancelaciones adicionales a cada observable calculado,
- y el ajuste a datos no depende de una inestabilidad radiativa inmediata.

### Naturalidad fuerte (objetivo de completion)
La TUO se considerará fuerte en naturalidad solo si se identifica una simetría o mecanismo que proteja:

- \(\nu^2\) frente a correcciones cuadráticas,
- \(V_0\) frente a correcciones cuárticas,
- y la estructura del acoplamiento oscuro frente a operadores no deseados.

WP8 congela que la TUO **todavía no cumple** este criterio fuerte.

---

## Estrategias de protección simétrica bajo consideración

WP8 congela como candidatas teóricas para etapas futuras:

### P8.1 — Simetría de desplazamiento aproximada
\[
\phi \to \phi + c
\]

rota suavemente por \(V(\phi)\) y el acoplamiento oscuro.

### P8.2 — Simetría de escala / conforme
En una completion más profunda, el tejido podría heredar una simetría de escala cuya ruptura genere \(V_0\) y \(\nu^2\).

### P8.3 — Mecanismos de sequestering o ajuste dinámico del vacío
WP8 congela esta línea como posible solución futura al problema de \(V_0\), pero no como parte ya demostrada de la TUO actual.

---

## Validaciones mínimas exigidas por WP8

### V8.1 — Coherencia EFT
La EFT debe operar en una ventana clara:

\[
H_0 \ll E \ll \Lambda_{\rm EFT}.
\]

### V8.2 — Estabilidad de parámetros
Las correcciones radiativas no deben destruir la región del espacio de parámetros que ajusta datos cosmológicos.

### V8.3 — No reintroducción de acoplamientos visibles prohibidos
Las correcciones cuánticas no deben generar un acoplamiento bariónico fuerte del tejido incompatible con la filosofía congelada en WP3.

### V8.4 — Compatibilidad con WP0–WP7
La discusión de estabilidad cuántica no debe contradecir la EFT cosmológica actualmente validada como marco de trabajo.

---

## Qué sí cierra WP8

WP8 cierra que:

1. la TUO actual se interpreta como una EFT cosmológica con dominio de validez finito;
2. la estabilidad radiativa de \(\nu^2,\lambda_4,\beta_0\) es un problema bien planteado y separable del sector observacional ya estudiado;
3. el problema de \(V_0\) se reconoce explícitamente como un problema abierto de naturalidad, no resuelto aún;
4. la TUO puede seguir siendo científicamente útil y válida como EFT incluso sin una solución completa inmediata al problema del vacío.

---

## Qué NO cierra todavía WP8

WP8 no cierra:

- una UV completion cuántica del tejido,
- una demostración completa de renormalizabilidad,
- una protección exacta del potencial frente a loops,
- ni una solución definitiva al problema de la constante cosmológica efectiva.

---

## Resultado conceptual congelado

La frase oficial de WP8 es:

> **La TUO actual es una EFT cosmológica cuánticamente aceptable en sentido débil si existe una ventana de validez donde sus parámetros permanezcan radiativamente estables; sin embargo, la naturalidad fuerte del potencial del tejido y del vacío efectivo sigue siendo un problema abierto, como en la mayoría de teorías modernas del sector oscuro.**

---

## Entregables WP8

### D8.1
Documento de criterios de estabilidad cuántica y naturalidad de la EFT TUO.

### D8.2
Tabla de parámetros que requieren protección por simetría.

### D8.3
Lista de estrategias de UV completion compatibles con la EFT cosmológica actual.

---

## Criterios de éxito

WP8 se considerará completado si, al final del paquete, queda fijado:

1. qué se exige para considerar radiativamente estable a la TUO como EFT;
2. qué parámetros pueden tratarse honestamente como EFT y cuáles requieren una explicación más profunda;
3. y qué tipo de UV completion sería compatible con la fenomenología cosmológica ya establecida.

---

## Siguiente paso lógico

Con WP8 congelado, el siguiente paquete natural es:

**WP9 — firma exclusiva y falsabilidad de la TUO**
