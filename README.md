# bag_ilropr

Author: Bob Zhou

This is a library for circuits for a multi-phase clock generation and rotation circuit based on multi-phase injection-locked ring oscillators.

## Circuit Hierarchy (outline)
- `TopV1`: AMS core block
    - `CoreOutDAC` - Core oscillator + C2C output buffers + IDACs
        - `Core` - Oscillator core
            - `CoreUnit` - Oscillator core unit cell
                - `CoreDiffPair` - diff pair array
                - `DiffRes` (bag3_analog)
        - `CML2CMOS_diff_AC` - Psuedodiff CML2CMOS - AC-coupled TIA
            - `CML2CMOS_AC` - Single-ended CML2CMOS
                - `mosres` - Stacked MOS resistor, for TIA feedback
                - `inv` (bag3_digital)
                - `mos_cap`
        - `IDACArray` - IDAC, with on / off switches
            - `IDACUnit`
                - `IDACCell`
    - `InBuff` - Input replica buffer + replica bias
        - `InBuffCore` - Input replica core
            - `HighPassFilter` (bag3_analog) - AC coupling the input
            - `BufCore` - replica swing conditioning
                - `BufUnit` - buffer unit cell. Generator can instantiate multiple.
                    - BufDiffPair
                    - DiffRes (bag3_analog)
            - `BufUnit` - replica biasing
        - `IDACArray` - IDAC, always on
            - `IDACUnit`
                - `IDACCell`

## Project dependencies:
- `bag3_testbenches` - base testbench library
- `bag3_analog` - resistors
- `bag3_digital` - logic cells (inverters, etc.)
