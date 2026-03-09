# WP5 — Implementación Einstein–Boltzmann completa de la TUO

## Estado del paquete
**Congelado como especificación técnica de referencia para la implementación lineal completa de la TUO.**

## Objetivo
Elevar la TUO desde una EFT cosmológica con módulos de background y growth comprimido a una **teoría lineal relativista completa**, capaz de producir de forma autoconsistente:

- anisotropías del CMB,
- lensing del CMB,
- funciones de transferencia,
- espectro lineal de materia,
- BAO full-shape,
- RSD lineal,
- y observables derivados como `fσ8(z)` y `S8`.

La meta es disponer de un motor tipo Einstein–Boltzmann comparable, en el límite adecuado, a una implementación de referencia tipo CLASS/CAMB, pero con la física propia de la TUO.

---

## 1. Alcance de WP5

### 1.1 Qué cubre
WP5 cubre el régimen:

- **lineal** en perturbaciones,
- **relativista**,
- para todos los componentes cosmológicos relevantes:
  - bariones,
  - fotones,
  - neutrinos,
  - CDM,
  - campo del tejido `phi`,
  - potenciales métricos.

### 1.2 Qué no cubre todavía
No forman parte del cierre de WP5:

- no linealidad fuerte,
- halo model,
- bias de galaxias no lineal,
- full-shape no lineal,
- EFT of LSS,
- ni objetos compactos/ondas gravitacionales.

Eso pertenece a paquetes posteriores.

---

## 2. Entrada teórica congelada desde WP0–WP4

La implementación lineal completa se basa en la EFT ya congelada:

```math
S_{\rm TUO}^{\rm eff}
=
\int d^4x\,\sqrt{-g}
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12 (\partial\phi)^2
-V(\phi)
\right]
+
S_b[g]
+
S_r[g]
+
S_c[e^{2\beta(\phi)}g].
```

con

```math
\beta(\phi)=\beta_0\phi/M_{\rm Pl},
\qquad
V(\phi)=V_0+\frac12\nu^2\phi^2+\frac{\lambda_4}{4}\phi^4.
```

El sector primordial congelado es:

- perturbaciones adiabáticas,
- semilla primordial estándar `A_s`, `n_s`,
- bloqueo radiativo natural del tejido.

---

## 3. Variables dinámicas que debe propagar el solver

WP5 congela como **conjunto mínimo de variables físicas lineales**:

### 3.1 Background

```math
\{a(\tau),\; \mathcal H(\tau),\; \phi(\tau),\; \phi'(\tau),\; \rho_b,\rho_c,\rho_\gamma,\rho_\nu\}
```

con `\tau` tiempo conforme y `\mathcal H = a'/a`.

### 3.2 Sector escalar lineal

En una formulación teórica de referencia (gauge newtoniano):

```math
\{\Phi,\Psi,\delta_b,\theta_b,\delta_c,\theta_c,\delta_\gamma,\theta_\gamma,
\delta_\nu,\theta_\nu,\sigma_\nu,\delta\phi,\delta\phi'\}
```

### 3.3 Sector tensorial
Por defecto, el sector tensorial se congela al de GR en esta etapa:

```math
h_{ij}^{\rm TT}
```

sin nuevos modos tensoriales propagantes del tejido en la EFT lineal de trabajo.

---

## 4. Gauge de referencia y gauge de implementación

### 4.1 Gauge teórico de referencia
WP5 adopta **gauge newtoniano** como referencia para escribir la teoría:

```math
ds^2 = a^2(\tau)\left[-(1+2\Psi)d\tau^2 + (1-2\Phi)\delta_{ij}dx^idx^j\right].
```

Esto se hace porque:

- el acoplamiento oscuro es más transparente,
- la interpretación física de `\Phi`, `\Psi` es directa,
- y el cierre covariante de la TUO es más limpio en esta base.

### 4.2 Gauge de implementación permitida
La implementación numérica puede realizarse en:

- gauge newtoniano, o
- gauge síncrono (si se injerta sobre CLASS),

pero **debe demostrar equivalencia gauge-invariante** en el límite `\beta_0 -> 0` y en observables finales.

---

## 5. Acoplamiento oscuro lineal congelado

Del WP3 se hereda el intercambio covariante:

```math
\nabla_\mu T^{\mu\nu}_{(c)} = -\alpha(\phi) T_{(c)} \nabla^\nu \phi,
\qquad
\alpha(\phi)=\beta_0/M_{\rm Pl}.
```

En background:

```math
Q = \alpha(\phi)\dot\phi\rho_c.
```

En variables usadas por la TUO:

```math
Q = \beta_0 H y \rho_c,
\qquad y = d(\phi/M_{\rm Pl})/dN.
```

### Congelación para el solver lineal
WP5 fija que el módulo lineal debe implementar:

1. continuidad de CDM modificada,
2. ecuación de Euler modificada,
3. fuente escalar `\delta\phi` y retroalimentación en las ecuaciones de Einstein,
4. sin recurrir a activaciones heurísticas tipo `tanh`.

---

## 6. Gravedad efectiva y dependencia de escala

WP5 congela que, como mínimo, el solver debe ser capaz de reproducir la forma efectiva subhorizonte:

```math
\mu(k,a) \equiv G_{\rm eff}(k,a)/G
=
1 + \frac{2\beta_0^2 k^2}{k^2 + a^2 m_{\rm eff}^2(a)}.
```

con

```math
m_{\rm eff}^2(a)=V_{,\phi\phi}.
```

### Interpretación
- si `k^2 >> a^2 m_eff^2`: la quinta fuerza está activa,
- si `k^2 << a^2 m_eff^2`: la teoría se apantalla y vuelve a GR.

### Requisito
El solver debe poder pasar de forma continua entre:

- régimen completo lineal relativista,
- y régimen subhorizonte efectivo.

---

## 7. Sistema lineal mínimo a implementar

### 7.1 Ecuaciones de Einstein linealizadas
El solver debe integrar las ecuaciones de Einstein modificadas por `delta phi`, garantizando consistencia con:

```math
\delta G_{\mu\nu} = M_{\rm Pl}^{-2} \delta T_{\mu\nu}^{\rm total}.
```

### 7.2 Ecuación lineal del tejido
Debe implementarse la ecuación perturbada del campo:

```math
\delta\phi'' + 2\mathcal H \delta\phi' + (k^2 + a^2 V_{,\phi\phi})\delta\phi = \text{fuentes métricas + fuente CDM}.
```

### 7.3 Componentes materiales
Deben propagarse, al menos:

- bariones,
- fotones,
- neutrinos,
- CDM acoplado,
- tejido.

---

## 8. Módulos computacionales requeridos

WP5 congela la arquitectura mínima del solver:

### Módulo A — Background
Calcula:

- `H(a)`
- `rho_i(a)`
- `phi(a)`
- `Q(a)`
- `r_d`, `z_drag`, `z_eq`

### Módulo B — Thermodynamics
Puede heredarse del solver estándar, pero usando el `H(a)` modificado de la TUO.

### Módulo C — Primordial
Recibe:

- `A_s`
- `n_s`
- `alpha_s`
- `r`

según la clausura congelada en WP4.

### Módulo D — Perturbations
Resuelve el sistema lineal relativista completo.

### Módulo E — Transfer
Produce funciones de transferencia para cada especie.

### Módulo F — Spectra
Genera:

- `C_l^{TT}`
- `C_l^{TE}`
- `C_l^{EE}`
- `C_l^{\phi\phi}`
- `P(k,z)`
- `fσ8(z)`

### Módulo G — Likelihood interface
Debe conectarse con likelihoods cosmológicos estándar.

---

## 9. Estrategia de implementación recomendada

WP5 congela como estrategia preferida:

> **fork de un solver Einstein–Boltzmann existente (preferiblemente CLASS) con un módulo TUO específico.**

### Razones
- minimiza errores en física estándar,
- permite validación directa en el límite `LambdaCDM`,
- reutiliza termodinámica, neutrinos, reionización y proyecciones angulares ya probadas.

### Implementación alternativa permitida
Un solver propio desde cero es permitido, pero solo si reproduce primero el caso `LambdaCDM` con precisión comparable al código de referencia.

---

## 10. Validación obligatoria de WP5

### 10.1 Validación de background
En el límite `\beta_0 -> 0` y `\phi` congelado, el solver debe reproducir:

- `H(z)`
- `r_d`
- `z_eq`
- `D_M, D_H, D_V`

con diferencias relativas pequeñas respecto al solver estándar.

### 10.2 Validación de crecimiento
Debe reproducir en el límite estándar:

- `D(z)`
- `f(z)`
- `fσ8(z)`
- `σ8_0`

con errores relativos pequeños y controlados.

### 10.3 Validación de CMB
En el límite `LambdaCDM`, los espectros angulares deben satisfacer un criterio de consistencia del tipo:

```math
\max_\ell \left|\frac{C_\ell^{\rm TUO}-C_\ell^{\rm ref}}{C_\ell^{\rm ref}}\right| < \epsilon_{\rm CMB}
```

con `\epsilon_CMB` fijado por el proyecto en fase de implementación.

### 10.4 Validación gauge
Si la implementación usa gauge síncrono, debe verificarse equivalencia con observables gauge-invariantes respecto a la formulación teórica de referencia.

---

## 11. Observables que WP5 debe producir

WP5 se considera funcional solo si produce de manera estable y reproducible:

- `H(z)`
- `D_M(z), D_H(z), D_V(z)`
- `r_d`
- `z_drag`
- `z_eq`
- `P(k,z)`
- `fσ8(z)`
- `σ8_0`
- `S8`
- `C_l^{TT}`
- `C_l^{TE}`
- `C_l^{EE}`
- `C_l^{\phi\phi}` (lensing CMB)

---

## 12. Entregables congelados para WP5

### D5.1
**Solver Einstein–Boltzmann lineal de la TUO** (fork o módulo propio)

### D5.2
**Suite de validación TUO ↔ LambdaCDM**

### D5.3
**Interfaz de likelihood completa** para:

- CMB,
- BAO full-shape / comprimido,
- SN,
- RSD / growth,
- weak lensing.

### D5.4
Documento técnico:
**“Implementación lineal Einstein–Boltzmann de la TUO”**

---

## 13. Criterio de éxito de WP5

WP5 se considera completado si y solo si:

1. el solver reproduce correctamente `LambdaCDM` como límite,
2. produce todos los observables lineales relevantes,
3. la TUO puede confrontarse con CMB y large-scale structure a igualdad técnica con el modelo estándar,
4. y la implementación es estable numéricamente en el espacio de parámetros viable.

---

## 14. Riesgos reconocidos

### Riesgo A
Que la teoría, ya al nivel Einstein–Boltzmann, quede forzada hacia el rincón casi-`LambdaCDM`.

### Riesgo B
Que la TUO muestre tensiones fuertes con CMB completo o full-shape growth.

### Riesgo C
Que aparezcan problemas gauge o de estabilidad numérica no vistos en el régimen comprimido.

Estos riesgos son aceptados como parte normal del programa de prueba rigurosa de la teoría.

---

## 15. Resultado conceptual congelado

La frase oficial de WP5 es:

> **La TUO debe ser elevada desde una EFT cosmológica comprimida a una teoría lineal relativista completa mediante un solver Einstein–Boltzmann que reproduzca LambdaCDM en el límite adecuado y permita una confrontación observacional de primera división.**

---

## 16. Relación con el paquete siguiente

Una vez completado WP5, la TUO podrá abordarse en:

- weak lensing,
- `S8`,
- CMB lensing,
- growth full-shape,
- y tests de tensión cosmológica reales.

Por tanto, el siguiente paquete lógico tras WP5 es:

**WP6 — growth, weak lensing y tensión S8.**
