# Qualification sprint verification

Date: 20 November 2026, Kazakhstan.

## Audit and priorities

1. Preserve the original Streamlit project and the four still-image workflows. Located at `D:\Ular\Documents\AEROX-AI`. No files in that source tree were intentionally edited. The duplicate qualification folder contains the new UI and checkpoint copies.
2. Use the existing person detector to process MP4 on GPU and CPU. Confirm the original metrics and export a reproducible result.
3. Carry forward the ExG baseline, then implement grid comparison and simulation. Exercise all three Streamlit pages before assembling submission materials.
4. Capture the genuine UI on the development PC. Use cleared media for public evidence, check the new 2-page AI Card and verify both public video links.

## Test coverage

| Module | Tested behavior | Open item and next action |
|---|---|---|
| GUARD | Actual RTX 5060 and CPU inference, 12 analyzed frames each, timestamp/frame accounting, 51 peak and 348 per-frame boxes counted across samples on both devices, evidence selection and report control. Malformed MP4 is rejected. | Test with authorized native drone MP4 and capture a continuous browser recording. Source rights and independent detection quality remain To Be Verified. |
| AGRI | 53.125% on the exact fixture, 0% and 100% uniform cases, full pixel accounting on a non-divisible grid, visible yield status, report control and readable exported ZIP. | Analyze an approved RGB farmland photo. Field accuracy and crop-specific yield signals require separate reference data. |
| FLIGHT | Streamlit AppTest verified both routes, 5 and 7 waypoints, checklist gating, simulated completion and reset of displayed progress when a check is cleared. | Record the virtual route and confirm the exact physical F405 model, firmware and available telemetry before bench work. |

The Streamlit AppTest assertions ran against the app with an in-process event-loop substitute for the test runner because this execution environment blocks local TCP connections. The application code does not use that substitute. Production browser rendering, actual screenshot capture, downloaded attachment from a browser and continuous screen recording could not be checked here. The export archives themselves were verified and contain actual outputs.

Demo footage and screenshots: To Be Verified. Public media clearance: To Be Verified. Portal submission confirmation: To Be Verified.



Update — 20 November: The team captured genuine browser screenshots of all three modules. See BROWSER_DEMO.md for the observations, scope and remaining recording work.
