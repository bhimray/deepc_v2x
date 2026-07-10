# Research-Grade CAM / DeePC Data Work Log

This document tracks the sequential changes being made to make the NR-V2X NGSIM
scenario produce research-grade data suitable for DeePC Hankel matrices.

## Goal

The scenario should produce physically meaningful, repeatable, and traceable
time-series data for DeePC identification/control. In particular, the logged
packet-level data used to compute PRR, PIR, delay, CBR, and traffic context must
not depend on non-unique or standard-wrapped CAM fields.

## Completed Step 1: Separate KPI Sequence From CAM ASN.1 Payload

File changed:

- `scratch/nr_v2x_ngsim_deepc.cc`

### Problem Fixed

The previous implementation used CAM `generationDeltaTime` as the packet
sequence identifier for KPI logging.

That is not research-grade because `generationDeltaTime` is an ETSI CAM timing
field, not a unique packet identifier. It is computed modulo 65536 ms and wraps
every 65.536 seconds. In longer runs this can cause TX/RX matching collisions,
wrong PRR numerator counts, wrong PIR pairing, and wrong delay estimates.

### Implementation

A new ns-3 packet tag, `KpiPacketTag`, was added inside the scratch scenario.
It carries simulator-only metadata:

- `sequence`: monotonic `uint64_t` per transmitting CAM app
- `txTimeNs`: simulator transmit timestamp in nanoseconds

The CAM payload remains ASN.1 UPER encoded and still carries the standard CAM
`generationDeltaTime`. The KPI sequence is outside the ASN.1 payload and is used
only by the simulator/logger.

### Data Impact

The TX/RX packet logs now use a monotonic KPI sequence instead of wrapped
`generationDeltaTime`.

Affected logs:

- `*_tx_packet_log.csv`
- `*_rx_packet_log.csv`

This improves:

- TX/RX matching stability
- PRR window correctness
- duplicate-RX filtering
- PIR pair tracking
- packet delay calculation
- long-run suitability for DeePC Hankel matrix construction

### Delay Calculation Change

Previous delay estimate:

- RX time minus wrapped CAM `generationDeltaTime`

Current delay estimate:

- RX simulator timestamp minus `KpiPacketTag::txTimeNs`

This avoids ambiguity after `generationDeltaTime` wraparound.

### Additional Fix

A decoded CAM memory leak was fixed in the RX path. If IP-to-node resolution
fails after ASN.1 decode, the decoded CAM structure is now freed before return.

## Initial Verification Note

Build command used:

```bash
CCACHE_DISABLE=1 ./ns3 build nr_v2x_ngsim_deepc
```

The first sandboxed build failed due to `ccache: error: Read-only file system`.
Running with `CCACHE_DISABLE=1` allowed the build to progress into the automotive
module.

At this point in the sequence, the build did not complete because of existing
automotive module integration errors outside the KPI-sequence change. Those
blockers were handled in later steps.

Observed blockers at that checkpoint:

- `src/automotive/model/Facilities/denBasicService.cc`: `TimestampTag` API
  mismatch in `timestamp.Get()`.
- `src/automotive/model/Facilities/caBasicService.cc`: `Ptr<LDM>` is compared
  with `NULL`; newer ns-3 expects `nullptr`-style checks.

## Completed Step 2: Fix Current Automotive Compile Blockers

Files changed:

- `src/automotive/model/Facilities/caBasicService.cc`
- `src/automotive/model/Facilities/caBasicService_v1.cc`
- `src/automotive/model/Facilities/cpBasicService.cc`
- `src/automotive/model/Facilities/cpBasicService_v1.cc`
- `src/automotive/model/Facilities/denBasicService.cc`
- `src/automotive/model/Facilities/denBasicService_v1.cc`
- `src/automotive/model/Facilities/iviService.cc`
- `src/automotive/model/Facilities/LDM.cc`
- `src/automotive/model/Facilities/mcBasicService.cc`
- `src/automotive/model/Facilities/mcBasicService.h`
- `src/automotive/model/Facilities/VRUBasicService.cc`
- `src/automotive/model/Security/security.h`
- `src/automotive/model/ASN1/full-v1-v2/asn_system.h`
- `src/automotive/model/ASN1/full-v1-v2/McmBasicContainer.h`
- `src/automotive/model/ASN1/full-v1-v2/McmBasicContainer.c`

### Problem Fixed

The automotive module was failing before the scratch scenario could be compiled.
Two compatibility issues were fixed:

1. `TimestampTag` API mismatch

   The automotive facilities code called `timestamp.Get()`, but the build
   resolves `ns3/timestamp-tag.h` to ns-3's network `TimestampTag`, which exposes
   `GetTimestamp()` returning a `Time`. The facilities code now uses:

   ```cpp
   timestamp.GetTimestamp().GetSeconds()
   ```

   This preserves the existing `SetSignalInfo(...)` expectation of a timestamp
   represented as seconds.

2. `Ptr<LDM>` null comparisons

   Several facilities services compared `Ptr<LDM>` against `NULL`. With the
   active ns-3 `Ptr<>` operators, this fails template resolution. These checks
   were updated to compare against `nullptr`.

3. `Ptr<TraciClient>` null comparisons

   The next targeted build progressed into `LDM.cc` and found the same pattern
   for `m_client`, a `Ptr<TraciClient>`. Those checks were also updated from
   `NULL` to `nullptr`.

4. Missing security include

   `Security` stores received certificates in `std::map`, but `security.h` did
   not include `<map>`. The include was added.

5. C++20 reserved keyword in generated MCM ASN.1 code

   C++20 reserves `concept`, while the generated `McmBasicContainer` structure
   had a field named `concept`. The C/C++ member was renamed to `concept_`, and
   the generated descriptor `offsetof(...)` plus `mcBasicService` access were
   updated. The ASN.1 element name string remains `"concept"`, so this is a C++
   source compatibility fix, not a schema change.

### ASN.1 Compatibility Note

During build verification, generated ASN.1 code in
`src/automotive/model/ASN1/full-v1-v2/asn_system.h` treated C++ compilation as
the "old compiler" branch for `SIZE_MAX` / format macros. The C++ branch was
enabled by accepting `__cplusplus >= 201103L` alongside C99 detection.

This is a build-integration fix only. It does not change CAM/DENM ASN.1 schema
or encoded payload content.

## Completed Step 3: Finish Target Build Integration

Files changed:

- `src/automotive/model/TxTracker/txTracker.cc`
- `contrib/nr/model/nr-spectrum-phy.h`
- `contrib/nr/model/nr-spectrum-phy.cc`
- `src/automotive/model/GeoNet/geonet.cc`
- `src/automotive/model/ASN1/full-v1-v2/RegionalExtension.c`
- `src/automotive/CMakeLists.txt`

### Problem Fixed

The target then exposed several build-integration blockers after the facilities
compatibility fixes:

1. TxTracker used older Wi-Fi and NR private interference APIs.
2. GeoNet compared `Ptr<Socket>` against `NULL`.
3. Generated `RegionalExtension.c` contained an empty information-object-set
   array rejected by the active C compiler.
4. The automotive module linked an unavailable `${libcv2x}` module target.
5. Signal-info packet tag implementations were present on disk but omitted from
   the automotive CMake source list.

### Implementation

TxTracker now retrieves the Wi-Fi interference helper through the public
`InterferenceHelper` attribute and calls the current three-argument
`AddForeignSignal(...)` API.

`NrSpectrumPhy` now exposes:

```cpp
void AddExternalInterference(Ptr<const SpectrumValue> psd, Time duration);
```

The method injects externally computed interference PSD into NR data, control,
SRS, and sidelink interference calculators. TxTracker uses this method instead
of removed direct accessors.

GeoNet socket/supervisor null initialization and checks were updated to
`nullptr`.

`RegionalExtension.c` keeps the generated table row count at zero but uses one
dummy zero-initialized row so the C compiler no longer sees a zero-sized array.
The dummy row is unreachable because the information-object-set row count is
still `0`.

The automotive CMake link list no longer hard-links `${libcv2x}` because this
workspace does not currently build an active `cv2x` module target. The four
signal-info tag implementation files are now included in the automotive module:

- `model/SignalInfo/rsrp-tag.cc`
- `model/SignalInfo/rssi-tag.cc`
- `model/SignalInfo/sinr-tag.cc`
- `model/SignalInfo/size-tag.cc`

### Verification Status

The target now builds successfully with:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=1 CCACHE_DISABLE=1 ./ns3 build nr_v2x_ngsim_deepc
```

Notes:

- CMake still prints a ccache config warning during reconfigure, but the build
  completes.
- `src/wave` is skipped because it has no `CMakeLists.txt`.
- The successful result is build verification only. Runtime data validation is
  still pending.

## Current CAM Standard Status

The scratch scenario no longer hand-builds CAM ASN.1 messages directly. CAM
generation now goes through the automotive stack:

- `NgsimVehicleDataProvider` adapts ns-3 NGSIM mobility to the `VDP` interface.
- `CamApplication` owns `CABasicService`, `btp`, and `GeoNet` objects.
- `CABasicService` generates and receives CAMs.
- KPI metadata remains outside the CAM ASN.1 payload through `KpiPacketTag`.

This is the correct direction for standard CAM behavior because CAM field
construction is now delegated to the automotive facilities service instead of
scratch-local ASN.1 assembly.

Remaining standard-grade CAM gaps:

- Station ID still comes from ns-3 node ID rather than a formal ITS station ID
  allocation strategy.
- Latitude/longitude are derived from a fixed local origin, so geodesy must be
  validated against the intended NGSIM map reference.
- The `VDP` adapter currently fills several confidence/optional fields with
  unavailable/default values.
- DCC and adaptive CAM generation behavior must be validated in runtime logs.
- Low-frequency container, path history, RSU behavior, and special vehicle
  containers are not yet validated for the NGSIM NR-V2X scenario.
- The GeoNet/BTP path is used inside the scratch application, but end-to-end
  standard networking behavior still needs runtime validation.

## Next Sequential Task

The next task should be runtime/data validation, not another structural CAM
refactor yet:

1. Run a short deterministic `nr_v2x_ngsim_deepc` scenario.
2. Confirm TX/RX packet logs are generated.
3. Check KPI sequence uniqueness and TX/RX matching.
4. Check delay, PRR, PIR, CBR, and DeePC CSV column consistency.
5. Only after that, continue improving CAM standard completeness field by field.

## Planned Follow-Up Tasks

After the build is clean:

1. Validate CAM field units, ranges, and round-trip ASN.1 UPER encode/decode.
2. Validate generated DeePC dataset columns and Hankel readiness.
3. Run a short deterministic scenario and compare packet logs against KPI CSV
   aggregates.
4. Add stronger station-ID and geodesy configuration.
5. Validate DCC/adaptive CAM behavior under different CBR conditions.
