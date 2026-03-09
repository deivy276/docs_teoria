# WP10 — Tests observacionales de falsabilidad de la TUO

## Estado
Congelado como paquete de trabajo observacional del programa mínimo de la TUO.

## Objetivo
Traducir la estructura teórica fijada en WP0–WP9 a un conjunto explícito de **tests observacionales capaces de falsar la rama mínima interactuante de la TUO** o de demostrar que colapsa efectivamente al límite \(\Lambda\)CDM.

---

## 1. Alcance de WP10

WP10 no introduce nueva física. Su función es:

1. ejecutar tests de consistencia y discriminación entre ramas,
2. cuantificar si la TUO mínima aporta contenido observacional real,
3. establecer criterios claros de falsación, supervivencia o colapso al límite estándar.

WP10 se aplica sobre la EFT de referencia congelada en WP0 y sobre la rama mínima interactuante definida en WP9.

---

## 2. Hipótesis operativa

La rama mínima de la TUO debe distinguirse observacionalmente mediante una modificación correlacionada de:

- expansión tardía \(H(z)\),
- crecimiento lineal \(f\sigma_8(z)\),
- weak lensing (vía \(S_8\) y observables relacionados),

mientras mantiene:

- invisibilidad temprana del tejido,
- sector tensorial infrarrojo estándar,
- límite \(\Lambda\)CDM accesible.

---

## 3. Observable principal de discriminación

La rama mínima interactuante no se juzga por un único observable, sino por la **consistencia conjunta** de:

\[
\{H(z),\; D_M(z),\; D_H(z),\; r_d,\; f\sigma_8(z),\; S_8\}.
\]

La falsabilidad de la TUO se evaluará por su capacidad (o incapacidad) para describir simultáneamente:

1. background,
2. growth,
3. lensing,

sin colapsar a \(\beta_0\to 0\) y sin pagar un coste estadístico excesivo.

---

## 4. Tests observacionales congelados

### T10.1 — Test de sectores del acoplamiento
Comparar explícitamente las ramas:

- sector positivo: \(\beta_0 > 0\),
- sector negativo: \(\beta_0 < 0\),
- sector libre: \(\beta_0 \in [-\beta_{\max},\beta_{\max}]\).

#### Objetivo
Determinar si el signo del acoplamiento es:

- preferido,
- indiferente,
- o efectivamente colapsado a cero.

#### Métricas
- \(\Delta\chi^2\)
- \(\Delta\chi^2_{\rm eff}\)
- \(\Delta\mathrm{AIC}\)
- \(\Delta\mathrm{BIC}\)
- evidencia Bayesiana (si se implementa)

#### Criterio de lectura
- si ambos sectores dan resultados equivalentes: el signo no está determinado,
- si uno domina claramente: posible preferencia física por el signo del acoplamiento,
- si ambos colapsan a \(\beta_0\approx 0\): la rama interactuante pierde poder explicativo.

---

### T10.2 — Test de colapso efectivo a \(\Lambda\)CDM
Evaluar si la rama mínima interactuante añade poder explicativo real o si simplemente reproduce el límite estándar.

#### Diagnóstico principal
La rama TUO se considera **colapsada de facto al límite \(\Lambda\)CDM** si ocurre simultáneamente que:

\[
|\beta_0| \to 0,
\qquad
x_{\rm ini} \to 0,
\qquad
\Delta \mathrm{AIC} > 0,
\qquad
\Delta \mathrm{BIC} > 0.
\]

#### Resultado posible
- **No colapso:** la rama interactuante conserva una región preferida no trivial.
- **Colapso parcial:** la rama sobrevive pero no mejora el modelo estándar.
- **Colapso total:** la TUO mínima interactuante se reduce observacionalmente al límite \(\Lambda\)CDM.

---

### T10.3 — Test conjunto background + growth + lensing
El test más importante de WP10.

#### Observable combinado
\[
\mathcal L_{\rm tot} = \mathcal L_{\rm SN}\,\mathcal L_{\rm BAO}\,\mathcal L_{\rm RSD}\,\mathcal L_{\rm WL}.
\]

#### Objetivo
Verificar si la TUO puede modificar growth y lensing de forma coordinada sin romper el background.

#### Preguntas clave
1. ¿La TUO mejora la consistencia entre \(f\sigma_8\) y \(S_8\)?
2. ¿El ajuste a weak lensing destruye SN+BAO?
3. ¿La teoría reduce la tensión de \(S_8\) sin desplazar patológicamente \(H(z)\) o \(r_d\)?

---

### T10.4 — Plano \((\Omega_m, S_8)\)
Se congela como test visual y cuantitativo obligatorio.

#### Motivación
La rama mínima de la TUO puede desplazar la predicción teórica en el plano:

\[
S_8 = \sigma_8\sqrt{\Omega_m/0.3}.
\]

#### Criterio
WP10 exige producir y comparar:

- banda \(\Lambda\)CDM,
- banda TUO sector positivo,
- banda TUO sector negativo,
- contornos observacionales de lensing.

#### Interpretación
- Si la TUO entra naturalmente en los contornos de weak lensing sin perder BAO/RSD, gana interés físico real.
- Si no mejora ese plano, pierde una de sus posibles ventajas más importantes.

---

### T10.5 — Null test de anisotropía gravitacional
En la rama mínima actual se congela:

\[
\eta(k,a)=\frac{\Phi}{\Psi}=1,
\qquad
\Sigma(k,a)=\mu(k,a).
\]

#### Implicación
Una detección robusta de:

\[
\eta(k,a)\neq 1
\]

falsaría directamente la rama mínima actualmente congelada, aunque no necesariamente toda la TUO.

#### Implementación observacional recomendada
- combinación lensing + clustering,
- observables tipo \(E_G\),
- reconstrucción conjunta de \(\mu\) y \(\Sigma\).

---

### T10.6 — Test de forma de \(f\sigma_8(z)\)
La TUO podría generar una modificación sutil de la curvatura o amplitud de \(f\sigma_8(z)\).

#### Criterio
Comparar la banda teórica de la TUO contra:

- una banda \(\Lambda\)CDM de referencia,
- compilaciones RSD,
- y observables reconstruidos de growth.

#### Firma buscada
No se busca una desviación arbitraria, sino una deformación **correlacionada** con la misma dinámica que gobierna el background.

---

### T10.7 — Test de naturalidad observacional
No basta con encontrar un mejor ajuste bruto. La rama interactuante debe evitar soluciones que:

- requieran \(h\) demasiado desplazado,
- requieran \(\omega_b\) o \(\omega_c\) fuera de priors razonables,
- o compren una pequeña mejora en \(\chi^2\) pagando un gran coste en priors.

#### Métricas obligatorias
- \(\chi^2_{\rm total}\)
- \(\chi^2_{\rm prior}\)
- \(\Delta\chi^2_{\rm eff}\)
- \(\Delta\mathrm{AIC}\)
- \(\Delta\mathrm{BIC}\)

---

## 5. Datasets mínimos de referencia

WP10 congela como combinación mínima de prueba:

1. SN tipo Ia (Pantheon+ o equivalente)
2. BAO comprimido de referencia
3. RSD / growth comprimido o covariante
4. weak lensing comprimido (cuando esté implementado)

Los tests de WP10 pueden iniciarse con likelihoods comprimidos, pero el criterio final del paquete exige una ruta clara hacia likelihoods más completos.

---

## 6. Resultados posibles permitidos por WP10

### Caso A — Supervivencia fuerte
La TUO mínima interactuante mejora o iguala a \(\Lambda\)CDM con penalización controlada y muestra una región no trivial con:

\[
\beta_0 \neq 0
\]

estable y bien convergida.

### Caso B — Supervivencia débil
La TUO reproduce bien los datos, pero no mejora a \(\Lambda\)CDM una vez penalizada la complejidad.

### Caso C — Colapso al límite estándar
La teoría total sobrevive, pero la rama interactuante no muestra necesidad observacional y colapsa a \(\Lambda\)CDM.

### Caso D — Falsación de la rama mínima
La rama mínima interactuante no puede acomodar growth+lensing sin entrar en conflicto con background o priors.

---

## 7. Criterios formales de falsación

La rama mínima interactuante de la TUO se considerará observacionalmente falsada si se cumple cualquiera de las siguientes condiciones de manera robusta y reproducible:

### F1. Falsación por lensing
Detección robusta de:

\[
\eta(k,a)\neq 1
\]

cuando la rama mínima congela \(\eta=1\).

### F2. Falsación por incompatibilidad conjunta
No existe región de parámetros que describa simultáneamente:

- SN,
- BAO,
- RSD,
- WL,

sin pagar un coste extremo en priors.

### F3. Falsación por colapso total
El ajuste completo exige:

\[
\beta_0\to 0,
	y \to 0,
	x_{\rm ini}\to 0,
\]

y además no aporta mejora de información.

### F4. Falsación tensorial externa
Detección de una modificación tensorial incompatible con la rama mínima congelada.

---

## 8. Observables obligatorios de salida de WP10

Toda corrida observacional de WP10 debe producir al menos:

1. \(H(z)\)
2. \(D_M(z), D_H(z), D_V(z)\)
3. \(f\sigma_8(z)\)
4. \(\sigma_8\)
5. \(S_8\)
6. \(\Delta\chi^2\)
7. \(\Delta\chi^2_{\rm eff}\)
8. \(\Delta\mathrm{AIC}\)
9. \(\Delta\mathrm{BIC}\)
10. bandas sector positivo / negativo / libre en el plano \((\Omega_m,S_8)\)

---

## 9. Entregables de WP10

### D10.1
Tabla comparativa:

- \(\Lambda\)CDM
- TUO sector positivo
- TUO sector negativo
- TUO sector libre

con métricas de ajuste y complejidad.

### D10.2
Gráficos mínimos obligatorios:

- \(f\sigma_8(z)\) con bandas teóricas y datos,
- plano \((\Omega_m,S_8)\),
- corner plots sectoriales,
- evolución de \(H(z)\).

### D10.3
Documento de decisión:

**“Resultado observacional de la rama mínima de la TUO”**

afirmando si la rama queda:

- favorecida,
- viable pero no preferida,
- colapsada al límite \(\Lambda\)CDM,
- o falsada.

---

## 10. Resultado conceptual congelado

La frase oficial de WP10 es:

> **La rama mínima interactuante de la TUO debe juzgarse por su capacidad para describir de forma correlacionada la expansión, el crecimiento y el weak lensing sin colapsar al límite \(\Lambda\)CDM ni pagar un coste de complejidad injustificable.**

---

## 11. Criterio final de cierre del programa mínimo

El programa mínimo TUO quedará observacionalmente cerrado si, al final de WP10, se puede responder con claridad a estas tres preguntas:

1. ¿Existe una región no trivial con \(\beta_0\neq 0\) que sobreviva a background + growth + lensing?
2. ¿Esa región mejora o al menos iguala a \(\Lambda\)CDM con una penalización aceptable?
3. ¿La rama mínima queda falsada, viable o colapsada al límite estándar?

Si esas tres preguntas tienen respuesta cuantitativa, la rama mínima de la TUO queda científicamente madura.
