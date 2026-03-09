# WP6 — Growth, weak lensing y tensión S8 en la TUO

## Estado del paquete
**Congelado como especificación técnica y observacional de referencia para el sector de crecimiento y lensing de la TUO.**

## Objetivo
Extender la EFT cosmológica de la TUO desde el régimen de background + growth lineal comprimido hacia una confrontación observacional más fuerte con:

- crecimiento de estructuras,
- weak lensing cósmico,
- y el parámetro combinado
  \[
  S_8 \equiv \sigma_8\sqrt{\Omega_m/0.3}.
  \]

La meta del paquete es responder, de forma falsable y sin tuning ad hoc, si la TUO puede:

1. reproducir el growth lineal observado,
2. aliviar o no la tensión en `S8`,
3. y mantener simultáneamente un buen ajuste a SN + BAO + universo temprano.

---

## 1. Alcance de WP6

### 1.1 Qué cubre
WP6 cubre el régimen:

- lineal y cuasi-lineal,
- observables de crecimiento y lensing débiles,
- comparación con likelihoods comprimidos o 2pt statistics,
- consistencia conjunta con la EFT congelada en WP0.

### 1.2 Qué no cubre todavía
No forman parte del cierre de WP6:

- full-shape no lineal completo de galaxias,
- modelado detallado de baryonic feedback,
- neutrinos masivos variables,
- reionización completa del CMB,
- ni objetos compactos / sector tensorial.

---

## 2. Entrada teórica congelada desde WP0–WP5

WP6 asume como base la EFT cosmológica de referencia:

```math
S_{\rm TUO}^{\rm eff}
=
\int d^4x\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\partial\phi)^2
-V(\phi)
\right]
+S_b[g]+S_r[g]+S_c[e^{2\beta(\phi)}g].
```

con:

```math
\beta(\phi)=\beta_0\phi/M_{\rm Pl},
\qquad
V(\phi)=V_0+\frac12\nu^2\phi^2+\frac{\lambda_4}{4}\phi^4.
```

y con el acoplamiento oscuro covariante ya congelado en WP3:

```math
Q = \beta_0 H y\,\rho_c,
\qquad y = d(\phi/M_{\rm Pl})/dN.
```

El growth lineal debe heredar la versión calibrada 5C.1.1 o superior, es decir:

- límite \(\Lambda\)CDM validado,
- normalización primordial consistente con \(A_s\),
- y ausencia de activaciones fenomenológicas tipo `tanh`.

---

## 3. Observables físicos congelados en WP6

WP6 congela como observables primarios:

### 3.1 Growth

- factor de crecimiento \(D(z)\),
- tasa de crecimiento
  \[
  f(z)=\frac{d\ln D}{d\ln a},
  \]
- observable RSD:
  \[
  f\sigma_8(z).
  \]

### 3.2 Weak lensing

- potencial de lente (Weyl):
  \[
  \Phi_W = \frac{\Phi+\Psi}{2},
  \]
- espectro de lente convergente:
  \[
  C_\ell^{\kappa\kappa},
  \]
- correlaciones angulares de shear:
  \[
  \xi_+(\theta),\qquad \xi_-(\theta),
  \]
- combinaciones tomográficas si el solver lo permite.

### 3.3 Parámetro combinado

- tensión de estructura:
  \[
  S_8=\sigma_8\sqrt{\Omega_m/0.3}.
  \]

---

## 4. Relación entre gravedad efectiva y lensing

WP6 congela que la primera implementación debe usar la modificación efectiva ya presente en la EFT:

```math
\mu(k,a) \equiv \frac{G_{\rm eff}(k,a)}{G}
=
1+\frac{2\beta_0^2 k^2}{k^2+a^2m_{\rm eff}^2(a)}.
```

con

```math
m_{\rm eff}^2(a)=V_{,\phi\phi}.
```

### 4.1 Hipótesis de primer cierre
En la primera versión de WP6 se congela:

```math
\eta(k,a) \equiv \Phi/\Psi = 1
```

en ausencia de evidencia interna fuerte de slip gravitacional en la EFT mínima.

Entonces el sector de lensing queda gobernado por:

```math
\Sigma(k,a) = \mu(k,a).
```

### 4.2 Extensión futura
Si la implementación Einstein–Boltzmann completa revela anisotropic stress efectivo del tejido, WP6 deberá promocionar:

```math
\Sigma(k,a)=\frac{\mu(k,a)}{2}\left(1+\eta(k,a)\right).
```

---

## 5. Programa observacional de WP6

### 5.1 Etapa A — likelihoods comprimidos
Usar:

- compilaciones de \(f\sigma_8(z)\),
- constraints comprimidos de `S8`,
- y comparaciones rápidas en el plano \((\Omega_m,S_8)\).

Objetivo:

- testar si la TUO puede aliviar la tensión de estructura,
- sin necesidad aún de 2pt functions completas.

### 5.2 Etapa B — weak lensing 2pt
Implementar comparación con:

- cosmic shear,
- galaxia–lente si es posible,
- kernels tomográficos,
- y priors de nuisances mínimos (alineamientos intrínsecos, sesgo multiplicativo, etc.).

### 5.3 Etapa C — forecast Euclid / Stage IV
Una vez fijada la dinámica de growth y lensing, producir previsiones para:

- Euclid-like shear,
- DESI + Euclid conjuntas,
- y sensibilidad a \(\beta_0\), \(\nu^2\), \(x_{\rm ini}\).

---

## 6. Datasets de referencia congelados para WP6

### 6.1 Datasets de crecimiento (exploratorio → serio)
Orden recomendado de uso:

1. compilaciones diagonales exploratorias de \(f\sigma_8\),
2. likelihoods comprimidos con covarianza homogénea,
3. RSD comprimido coherente,
4. full-shape en paquete posterior.

### 6.2 Datasets de weak lensing
WP6 congela como objetivos de referencia:

- constraints comprimidos de \(S_8\),
- likelihoods tipo cosmic shear 2pt,
- y, en la fase fuerte, encuestas Stage III/IV.

No se congela aún una colaboración específica como única fuente normativa; el criterio es:

> usar datasets públicos, con covarianza disponible y documentación suficientemente clara para reproducibilidad.

---

## 7. Preguntas científicas que WP6 debe responder

### Q1.
¿La TUO puede reducir \(S_8\) sin destruir BAO + SN + CMB temprano?

### Q2.
¿El ajuste del growth favorece:

- \(\beta_0\approx 0\),
- \(|\beta_0|>0\),
- o un signo concreto de \(\beta_0\)?

### Q3.
¿La TUO introduce una firma distintiva en:

- \(f\sigma_8(z)\),
- \(S_8\),
- o en el plano \((\Omega_m,S_8)\)?

### Q4.
¿La teoría mejora tensiones internas del modelo estándar o solo reproduce \(\Lambda\)CDM con más parámetros?

---

## 8. Criterios de éxito de WP6

WP6 se considera exitoso si se alcanza al menos uno de estos tres escenarios:

### Escenario A — ventaja física real
La TUO reduce la tensión en \(S_8\) y mantiene ajuste competitivo a background.

### Escenario B — equivalencia controlada
La TUO no mejora significativamente \(\Lambda\)CDM, pero demuestra que puede reproducir simultáneamente:

- background,
- growth,
- y weak lensing,

como caso generalizado consistente.

### Escenario C — falsación útil
La inclusión de weak lensing empuja a la TUO de forma robusta al rincón casi-\(\Lambda\)CDM o la tensiona severamente. Incluso en ese caso, WP6 sigue siendo exitoso, porque establece el límite real del modelo.

---

## 9. Riesgos científicos congelados

### Riesgo 1
Que la TUO solo mejore growth a costa de violar priors o degradar BAO.

### Riesgo 2
Que la modificación de growth no se traduzca en mejora de \(S_8\), sino solo en un corrimiento de parámetros sin ganancia física.

### Riesgo 3
Que el sector de lensing requiera una función \(\Sigma(k,a)\) más rica que la hipótesis mínima \(\Sigma=\mu\).

### Riesgo 4
Que la teoría no muestre firma observacional exclusiva y termine siendo una generalización efectiva no preferida de \(\Lambda\)CDM.

---

## 10. Entregables congelados de WP6

### D6.1
Likelihood comprimido de growth + `S8` compatible con la EFT de referencia.

### D6.2
Módulo de weak lensing lineal/cuasi-lineal compatible con la TUO.

### D6.3
Comparación observacional conjunta:

```math
\text{SN} + \text{BAO} + \text{growth} + \text{WL}
```

### D6.4
Plano de predicción:

```math
(\Omega_m, S_8)
```

y bandas de \(f\sigma_8(z)\) para:

- \(\Lambda\)CDM,
- TUO sector positivo,
- TUO sector negativo.

---

## 11. Resultado conceptual congelado

La frase oficial de WP6 queda fijada como:

> **La TUO debe ser juzgada no solo por su background cosmológico, sino por su capacidad de describir simultáneamente el crecimiento de estructuras y el weak lensing, donde una física del tejido podría diferenciarse de \(\Lambda\)CDM.**

---

## 12. Dependencias de WP6

WP6 depende de:

- WP0: EFT de referencia,
- WP3: acoplamiento oscuro covariante,
- WP4: semilla primordial congelada,
- WP5: motor Einstein–Boltzmann lineal o equivalente.

Si WP5 no se completa plenamente, WP6 puede avanzar en modo comprimido, pero no cerrar definitivamente el caso observacional.

---

## 13. Decisión congelada

Se congela como estrategia de referencia:

1. comenzar con `fσ8 + S8` comprimidos,
2. pasar a weak lensing 2pt,
3. solo después escalar a full-shape / Stage IV.

Esto evita sobrecargar la teoría con complejidad observacional antes de cerrar la física lineal.
