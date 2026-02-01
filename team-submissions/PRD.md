# Product Requirements Document (PRD)

**Project Name:** LABS-Quantum-Optimizer
**Team Name:** Perseverance
**GitHub Repository:** https://github.com/YavuzSelimMeletlioglu/2026-NVIDIA

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Yavuz Selim Meletlioglu | @YavuzSelimMeletlioglu | @Yavuz |
| **GPU Acceleration PIC** (Builder) | Berk Berat Turan | @BerkBerat | @berkberat |
| **Quality Assurance PIC** (Verifier) | Ozan Bilmez | @oziwaNNN | @oziwan |
| **Technical Marketing PIC** (Storyteller) | Aytuğ Şahinkanat | @AytugSahinkanat | @aytugsahinkanat |
| **Technical Marketing PIC** (Storyteller) | Hasan Zafer Bilir | @Hasan1Zafer | @ZaferTheKnows |

---

## 2. The Architecture

**Owner:** Project Lead

### Choice of Quantum Algorithm

**Algorithm:**  
Quantum Approximate Optimization Algorithm (QAOA) with a hardware-efficient ansatz

**Motivation:**  
- **Metric-driven:** We chose QAOA because its layer depth can be tuned to match the correlation length in LABS sequences, potentially improving approximation quality.  
- **Skills-driven:** This approach allows team members to deepen their understanding of variational quantum algorithms and parameter tuning while providing a clear comparison to the Counteradiabatic method from Phase 1.

### Literature Review

**Reference:**  
Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A Quantum Approximate Optimization Algorithm*. arXiv:1411.4028 [quant-ph]. [Link](https://arxiv.org/abs/1411.4028)

**Relevance:**  
Although LABS is not MaxCut, this paper shows how parameter concentration in QAOA can speed up optimization. We aim to replicate similar behavior in the LABS problem to enhance our quantum-seeded classical workflow.

---

## 3. The Acceleration Strategy (QAOA)

### Quantum Acceleration (CUDA-Q)

Quantum simulation inherently involves manipulating large state vectors. For N qubits, the state vector contains 2^N complex numbers; for example, at N=20, this corresponds to over one million numbers, and at N=30, over one billion. Every quantum gate operation requires matrix-vector multiplication across the entire vector. CPUs, with only a limited number of cores, struggle to handle such workloads efficiently. GPUs, on the other hand, are specifically designed for massively parallel computation, with thousands of cores and high memory bandwidth, allowing these operations to be performed orders of magnitude faster. Therefore, GPU acceleration is essential for practical quantum simulations. 

CUDA-Q enables this acceleration seamlessly by compiling quantum kernels through MLIR to NVIDIA’s optimized cuQuantum backend. Switching from CPU to GPU is as simple as calling `cudaq.set_target("nvidia")`, and for very large simulations, `cudaq.set_target("nvidia-mgpu")` allows the state vector to be distributed across multiple GPUs. We expect the speedup from GPU acceleration to scale significantly with problem size: roughly 10× for 10 qubits, 60× for 20 qubits, and over 100× for 25 qubits and beyond.

### Classical Acceleration (MTS)

For the classical portion, the standard Memetic Tabu Search evaluates neighbor solutions sequentially, which becomes a bottleneck. By replacing NumPy with CuPy, we can evaluate large batches of neighbor flips in parallel on the GPU, keeping the classical search synchronized with the accelerated quantum simulations. Together, this hybrid approach of GPU-accelerated quantum and classical computation maximizes hardware utilization and minimizes total solution time for the LABS problem.

### Hardware Targets

**Development and Production Environments:**  
We will develop and test logic on Qbraid’s CPU backend, perform initial GPU migration on a Brev L4 instance, and run final large-scale benchmarks on a Brev A100-80GB GPU.

---

## 4. The Verification Plan

**Owner:** Quality Assurance PIC

---

### Unit Testing Strategy

- **Framework:** `pytest` / `unittest`

- **AI Hallucination Guardrails:**

  - **Automated Regression:**  
    All AI-generated code is subjected to a 10-point validation suite covering Physics, Classical Search, and Quantum Integration.

  - **Zero-Tolerance for Energy Inconsistency:**  
    GPU-accelerated kernels are cross-verified against hand-calculated *Golden Samples* to ensure no artificial energy drops are introduced by floating-point errors.

  - **Safety-First Integration:**  
    Hamiltonian generators are tested for index-out-of-bounds safety before being deployed to the CUDA-Q backend.

---

### Core Correctness Checks

- **Check 1 (Symmetry Invariance):**

  - **Logic:**  
    LABS energy must remain identical under bit-flip ( \( S \rightarrow -S \) ) and reversal ( \( S \rightarrow S_{rev} \) ).

  - **Result:**  
    **PASS.** Verified with random sequence (Energy: 85) across all symmetry transformations.

---

- **Check 2 (Ground Truth – Small N):**

  - **Validation:**  
    \( N = 3 \) sequence `[1, 1, -1]` is mathematically proven to have energy `1.0`.

  - **Result:**  
    **PASS.** Got exact match `1.0`, confirming the precision of the core energy engine.

---

- **Check 3 (Search Monotonicity):**

  - **Logic:**  
    Classical optimization (Tabu Search) must strictly improve or maintain the current solution.

  - **Result:**  
    **PASS.** Significant energy drop observed (`135 → 55`) without violating monotonic constraints.

---

- **Check 4 (Hardware Index Safety):**

  - **Logic:**  
    Every term in the \( G_2 \) and \( G_4 \) Hamiltonian interaction graphs must map to a valid qubit index \( i < N \) to prevent memory corruption in CUDA-Q.

  - **Result:**  
    **PASS.** Index safety verified for all 20 \( G_2 \) terms and 50 \( G_4 \) unique indices.
    
---

## 5. Execution Strategy & Success Metrics

**Owner:** Technical Marketing PIC

### Agentic Workflow

**Plan:**  
We will orchestrate our AI and development tools as follows:  

1. **IDE & Coding:** Use Cursor as the primary IDE for coding and refactoring.  
2. **AI Assistance:** Use ChatGPT as an AI assistant to help with code suggestions, documentation, and test generation.  
3. **Knowledge Guardrails:** Maintain a `skills.md` file containing CUDA-Q documentation, QAOA references, and best practices to prevent AI hallucinations.  
4. **Verification Loop:** The QA Lead runs all unit tests (`tests.py`) on CPU first. If any test fails, the error logs are pasted back into the AI agent for automatic refactoring suggestions.  
5. **GPU Integration:** Once CPU tests pass, migrate the code to Brev L4/A100 GPUs for accelerated benchmarking, while monitoring resource usage closely.

---

### Success Metrics

**Metric 1 (Approximation Ratio):**  
Target ratio > 0.85 for N=20, aiming to show that QAOA seeds improve classical MTS performance compared to random initialization.

**Metric 2 (Speedup):**  
Expect at least 50× speedup in quantum kernel runtime for N=20 when switching from CPU simulation to GPU-accelerated CUDA-Q.

**Metric 3 (Scale):**  
Successfully run hybrid quantum-classical simulation for N up to 25 on GPU without memory overflow or performance degradation.

---

### Visualization Plan

**Plot 1:** `"Time-to-Solution vs. Problem Size (N)"` comparing CPU-only QAOA simulation vs. GPU-accelerated QAOA simulation.

**Plot 2:** `"Convergence Rate"` showing energy vs. iteration count for Quantum-Seeded MTS vs. Random-Seeded MTS, highlighting the benefit of QAOA seeding.

---

## 6. Resource Management Plan

**Owner:** GPU Acceleration PIC  

**Plan:**  
To efficiently manage our limited Brev credits while running QAOA and hybrid GPU-accelerated MTS, we will follow a staged approach:

1. **Development on CPU:**  
   All logic, unit tests, and initial algorithm debugging will be done on Qbraid CPU backend. This ensures correctness without consuming GPU credits.  

2. **Initial GPU Migration:**  
   Once the CPU implementation passes all tests, we will migrate to a **Brev L4 GPU instance** for initial testing and small-scale benchmarks. This allows us to validate GPU code and optimize kernel performance at low cost.  

3. **Large-Scale Benchmarking:**  
   Only for final, large N simulations (e.g., N ≥ 30), we will use a **Brev A100-80GB GPU instance**. Run time will be strictly monitored to avoid exceeding our credit limit.  

4. **Credit and Instance Management:**  
   - The GPU PIC is responsible for manually shutting down all instances whenever they are idle, during breaks, or overnight.  
   - Continuous monitoring of GPU usage will ensure that our $20 credit allocation is not exceeded.  
   - Estimated credit usage plan:  
     - Development & Testing on CPU: $0  
     - L4 GPU small-scale testing: 5 hours (~$5)  
     - A100 GPU final benchmarks: 4 hours (~$8)  
     - Buffer for unexpected runs: $4  

This approach ensures we maximize GPU efficiency for QAOA while controlling costs and avoiding “zombie instances.”

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| QAOA doesn't improve over DC | Medium | High | Document findings as "negative result" |
| GPU credits run out | Low | Critical | Strict resource management |
| Time runs out | Medium | High | Prioritize core deliverables |
| AI hallucinates code | High | Medium | Comprehensive test suite |

---

## 8. Timeline (Phase 2)

| Time | Task |
|------|------|
| Start | CPU validation on qBraid complete |
| +1 hour | Migrate to Brev, test GPU backend |
| +3 hours | Benchmark QAOA vs baseline |
| +5 hours | Classical MTS GPU acceleration |
| +7 hours | Final benchmarks and visualization |
| +8 hours | Presentation preparation |
| Deadline | Final submission |

---

*Document Last Updated: January 31, 2026*
