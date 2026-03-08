# WP0 — Congelación de la EFT de referencia de la TUO

## Estado
Aprobado como versión de referencia de trabajo.

## Objetivo
Fijar una única versión de la teoría efectiva cosmológica de la TUO para evitar bifurcaciones internas del proyecto. Toda simulación futura deberá declarar explícitamente si es:

1. compatible con esta EFT de referencia, o
2. una extensión/modificación de esta EFT.

---

## 1. Dominio de validez
La EFT de referencia de la TUO se considera válida para:

- cosmología homogénea e isótropa (FLRW),
- perturbaciones lineales subhorizonte,
- crecimiento lineal de estructuras,
- observables comprimidos de background y growth.

No se considera aún una completion fundamental del sector tensorial ni una implementación Einstein–Boltzmann completa.

---

## 2. Grados de libertad y campos
### Sector gravitacional
- Métrica física: `g_{mu nu}`.
- Grados de libertad tensoriales estándar de GR.

### Sector del tejido
- Campo efectivo cosmológico: `phi`.
- Interpretación: modo escalar infrarrojo del tejido del universo.

### Sector material
- Bariones: acoplamiento mínimo a `g_{mu nu}`.
- Radiación: acoplamiento mínimo a `g_{mu nu}`.
- Materia oscura fría (CDM): acoplamiento conforme al tejido.

---

## 3. Acción efectiva congelada
La acción efectiva de referencia es:

```math
S_TUO^eff = \int d^4x \sqrt{-g} \left[ \frac{M_{Pl}^2}{2}R - \frac12 (\nabla \phi)^2 - V(\phi) \right]
+ S_b[g,\psi_b] + S_r[g,\psi_r] + S_c[\tilde g^{(c)},\psi_c].
```

Con métrica efectiva del sector oscuro:

```math
\tilde g^{(c)}_{\mu\nu} = e^{2\beta(\phi)} g_{\mu\nu}.
```

Versión mínima congelada:

```math
\beta(\phi) = \beta_0 \, \phi / M_{Pl}.
```

### Potencial congelado
```math
V(\phi) = V_0 + \frac12 \nu^2 \phi^2 + \frac{\lambda_4}{4}\phi^4.
```

---

## 4. Ecuaciones cosmológicas de referencia
En FLRW plano:

```math
3 M_{Pl}^2 H^2 = \rho_b + \rho_r + \rho_c + \rho_\phi
```

```math
-2 M_{Pl}^2 \dot H = \rho_b + \frac43 \rho_r + \rho_c + \dot\phi^2
```

```math
\rho_\phi = \frac12 \dot\phi^2 + V(\phi), \qquad
p_\phi = \frac12 \dot\phi^2 - V(\phi)
```

```math
\ddot\phi + 3H\dot\phi + V_{,\phi} = \alpha(\phi) \rho_c,
\qquad \alpha(\phi)=\beta_0/M_{Pl}
```

```math
\dot\rho_c + 3H\rho_c = -\alpha(\phi)\dot\phi\rho_c
```

```math
\dot\rho_b + 3H\rho_b = 0,
\qquad
\dot\rho_r + 4H\rho_r = 0.
```

---

## 5. Principios físicos congelados
### 5.1 Bloqueo radiativo
Durante dominación radiativa:

```math
T_r = -\rho_r + 3P_r = 0
```

por lo que la radiación no fuentea directamente a `phi`. El campo queda congelado por fricción de Hubble en el universo temprano.

### 5.2 Límite LambdaCDM
La EFT debe reducirse a `LambdaCDM` cuando:

```math
\dot\phi \to 0, \qquad \beta_0 \to 0, \qquad V(\phi) \to V(\phi_*) = \Lambda_eff.
```

---

## 6. Sector lineal de crecimiento congelado
El growth lineal de referencia se modela con:

- acoplamiento dinámico proporcional a la evolución del campo,
- gravedad efectiva dependiente de escala.

### 6.1 Acoplamiento dinámico
```math
Q \propto \beta_0 \, \phi'(N) \, \rho_c,
\qquad N = \ln a.
```

### 6.2 Gravedad efectiva
```math
\mu(k,a) \equiv G_eff/G = 1 + \frac{2\beta_0^2 k^2}{k^2 + a^2 m_eff^2(a)}
```

con:

```math
m_eff^2(a) = V_{,\phi\phi}.
```

### 6.3 Normalización primordial
La referencia congelada para amplitud primordial es:

```math
A_s = 2.1 \times 10^{-9}
```

con predicción de `sigma8` derivada del módulo calibrado 5C.1.1, no perfilada libremente como parámetro independiente.

---

## 7. Convenciones de parámetros
### Parámetros físicos activos
- `beta0`: intensidad de acoplamiento oscuro
- `nu2`: masa efectiva del tejido
- `x_ini`: condición inicial del campo (convención usada en el pipeline)
- `h`
- `omega_b`
- `omega_c`

### Parámetros congelados por defecto
- `lambda4 = 0` salvo mención explícita
- sector bariónico y radiativo mínimos
- geometría espacial plana

---

## 8. Datasets de referencia congelados
### Background
- Pantheon+
- DESI DR2 BAO comprimido

### Growth exploratorio
- Gold2017 (`diag`)
- eBOSS DR16 comprimido con covarianza simplificada usada en el pipeline

**Nota:** Estos datasets de growth son válidos para exploración y calibración de la EFT, pero no reemplazan un likelihood full-shape oficial.

---

## 9. Criterios de validación congelados
La EFT se considera válida si satisface simultáneamente:

### 9.1 Límite LCDM del módulo de growth
- `max_rel_D < 1e-2`
- `max_rel_f < 1e-2`
- `sigma8_0_pred` del orden correcto (~0.8)

### 9.2 Salud cosmológica temprana
- `f_TUO(z=1100) << 1`
- `z_eq` en rango físicamente razonable (~3400)
- `r_d` en rango BAO aceptable (~148 Mpc)

### 9.3 Salud estadística mínima
- cadenas MCMC con `Rhat < 1.05` para todos los parámetros activos

---

## 10. Línea base observacional congelada
Resultado interpretativo de referencia:

- La TUO puede igualar o mejorar levemente el ajuste bruto a los datos.
- Una vez penalizada por complejidad, `LambdaCDM` sigue siendo generalmente preferida.
- La TUO sigue siendo viable como EFT cosmológica.
- La preferencia por `beta0 != 0` no está demostrada de forma robusta en este estado del proyecto.

---

## 11. Política de control de cambios
Toda modificación futura debe clasificarse como:

### Clase A — compatible con la EFT de referencia
No cambia:
- el contenido de grados de libertad efectivo
- la forma de la acción
- el mecanismo de acoplamiento

### Clase B — extensión EFT
Cambia alguno de:
- potencial `V(phi)`
- forma de `beta(phi)`
- sector de crecimiento lineal

### Clase C — beyond-EFT / completion
Cambia:
- contenido de grados de libertad
- simetría fundamental
- teoría tensorial subyacente

Toda simulación debe indicar explícitamente a qué clase pertenece.

---

## 12. Declaración de estado actual
La TUO queda congelada, en WP0, como:

> una EFT cosmológica escalar–tensorial con acoplamiento oscuro conforme, bloqueo radiativo natural, límite `LambdaCDM` bien definido y sector lineal de crecimiento calibrado físicamente.

No se afirma todavía una completion fundamental del universo.

---

## 13. Próximo paquete de trabajo
El siguiente WP recomendado, a partir de esta congelación, es:

- **WP1:** cierre cinemático del contenido de grados de libertad,
- seguido de **WP2:** prueba ghost-free.
