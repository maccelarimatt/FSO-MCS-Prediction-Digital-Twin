# Lab Testbed: Context-Aware FSO Bidirectional Link
**Project:** MSc Dissertation - Predictive MCS Selection for TVWS over FSO
**Researcher:** Matthew Maccelari (OCLab, Wits University)

---

## 1. Hardware Inventory
### Control & Processing
* **SDR:** 1x USRP B210 (2x2 MIMO, 70 MHz – 6 GHz).
* **Host PC:** Running Ubuntu/GNU Radio/Python (PyTorch for GRU).

### Optical Transmission (Forward Link: Central -> Remote)
* **Laser Driver:** Thorlabs **CLD1010** (Integrated Touch Screen Driver & TEC).
* **Laser Diode:** 660nm (Visible) or 1550nm (Pigtail).
* **Modulation:** Analog RF via CLD1010 'MOD IN' port (Back panel).

### Optical Transmission (Return Link: Remote -> Central)
* **Laser Driver:** Thorlabs **LDC210C** (Current) + **TED200C** (Temp).
* **Mixing:** Mini-Circuits **TB-510+ Bias-T** (Combines DC bias and RF signal).

### Reception & Optics
* **Photodetectors:** 2x Thorlabs **PDA10A2** (Si Fixed Gain, 150 MHz BW).
* **Mounts:** Thorlabs **CXY1A** XY Translating Lens Mounts.
* **Turbulence Sim:** Industrial Heat Gun (for scintillation).

---

## 2. Signal Path & Wiring
| Link Segment | Source Port | Destination Port | Signal Type |
| :--- | :--- | :--- | :--- |
| **Forward TX** | B210 TX/RX A | CLD1010 MOD IN | RF Voltage (TVWS) |
| **Forward RX** | PDA10A2 #1 Out | B210 RX2 B | Baseband Analog |
| **Return TX** | B210 TX/RX B | Bias-T (RF Port) | RF Voltage (TVWS) |
| **Return RX** | PDA10A2 #2 Out | B210 RX2 A | Baseband Analog |

---

## 3. Safety & Hardware Protection Protocols ⚠️
### A. Laser Safety (Eyesight)
* **Class 3B/3R Risks:** Do not look directly into the laser beam or its specular reflections (from mirrors/lenses).
* **Alignment:** Always perform initial alignment at the **lowest possible power** (just above threshold).
* **Stray Beams:** Use beam blocks (black cards) to terminate the beam at the end of the bench.

### B. Laser Diode Protection (Electrical)
* **Current Limits:** Always set the **Hard Current Limit** on the CLD1010 or LDC210C to $I_{max}$ specified in the diode datasheet. For 660nm diodes, this is often around $130\text{ mA}$.
* **ESD Warning:** Laser diodes are extremely sensitive to static. Use an ESD wrist strap when handling pigtails or plugging diodes into the driver sockets.
* **Reverse Bias:** Never connect the laser diode in reverse polarity; it will be destroyed instantly.

### C. SDR & RF Safety
* **TX Gain Limits:** The B210 can output $+10\text{ to }+17\text{ dBm}$. High RF power can saturate or damage the modulation input of the drivers. Start GNU Radio with **TX Gain = 0 dB** and increase slowly.
* **DC Block:** Ensure the Bias-T is oriented correctly. The DC current from the LDC210C must not flow back into the USRP TX port.

### D. Thermal & Optical Safety
* **Heat Gun:** Do not point the heat gun directly at plastic lens mounts or cables for extended periods. Keep it at a distance to create air turbulence without melting equipment.
* **Photodiode Saturation:** If the PDA10A2 output exceeds $5\text{ V}$, it is saturated. This won't damage it immediately, but it will clip your data. Use ND filters to attenuate the beam if needed.

---

## 4. Configuration & Constraints
* **Operating Frequency:** TVWS range (~470 MHz – 790 MHz).
* **Sample Rate:** Suggested 1-2 MSps (Stay within CLD1010 bandwidth).
* **Channel Awareness:** Real-time Scintillation Index ($\sigma_I^2$) and RSSI logged for GRU training.