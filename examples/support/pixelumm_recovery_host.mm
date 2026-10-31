#import <Foundation/Foundation.h>
#include <stdint.h>

// Advisory host metadata only. This does not touch Metal or wait for GPU work.
extern "C" int32_t gs_pixelumm_recovery_thermal_state(void) {
  return int32_t(NSProcessInfo.processInfo.thermalState);
}
