###############################################################################
################### MOOSE Application Standard Makefile #######################
###############################################################################
#
# Optional Environment variables
# MOOSE_DIR        - Root directory of the MOOSE project
#
###############################################################################
# Use the MOOSE submodule if it exists and MOOSE_DIR is not set
MOOSE_SUBMODULE    := $(CURDIR)/moose
ifneq ($(wildcard $(MOOSE_SUBMODULE)/framework/Makefile),)
  MOOSE_DIR        ?= $(MOOSE_SUBMODULE)
else
  MOOSE_DIR        ?= $(shell dirname `pwd`)/moose
endif

# Marlin must be a sibling of this application (override with MARLIN_DIR=...).
MARLIN_DIR ?= $(abspath $(CURDIR)/../marlin)

# Stop early if the dependency paths are incorrect.
ifeq ($(wildcard $(MOOSE_DIR)/framework/build.mk),)
  $(error Cannot find MOOSE framework/build.mk in MOOSE_DIR=$(MOOSE_DIR))
endif
ifeq ($(wildcard $(MARLIN_DIR)/src/base/MarlinApp.C),)
  $(error Cannot find Marlin src/base/MarlinApp.C in MARLIN_DIR=$(MARLIN_DIR))
endif

# framework
FRAMEWORK_DIR      := $(MOOSE_DIR)/framework
include $(FRAMEWORK_DIR)/build.mk
include $(FRAMEWORK_DIR)/moose.mk

################################## MODULES ####################################
# To use certain physics included with MOOSE, set variables below to
# yes as needed.  Or set ALL_MODULES to yes to turn on everything (overrides
# other set variables).

ALL_MODULES                 := no

CHEMICAL_REACTIONS          := no
CONTACT                     := no
ELECTROMAGNETICS            := no
EXTERNAL_PETSC_SOLVER       := no
FLUID_PROPERTIES            := no
FSI                         := no
FUNCTIONAL_EXPANSION_TOOLS  := no
GEOCHEMISTRY                := no
HEAT_TRANSFER               := no
LEVEL_SET                   := no
MISC                        := no
NAVIER_STOKES               := no
OPTIMIZATION                := no
PERIDYNAMICS                := no
PHASE_FIELD                 := no
POROUS_FLOW                 := no
RAY_TRACING                 := no
REACTOR                     := no
RDG                         := no
SHIFTED_BOUNDARY_METHOD     := no
SOLID_MECHANICS             := yes
STOCHASTIC_TOOLS            := no
THERMAL_HYDRAULICS          := no
XFEM                        := no

include $(MOOSE_DIR)/modules/modules.mk
###############################################################################

# ---------------------------------------------------------------------------
# Dependency application: compile/link libmarlin (no second Marlin executable).
# app.mk accumulates header include directories and app_LIBS for the next app.
# Do NOT include $(MARLIN_DIR)/Makefile here.
# ---------------------------------------------------------------------------
APPLICATION_DIR    := $(MARLIN_DIR)
APPLICATION_NAME   := marlin
BUILD_EXEC         := no
GEN_REVISION       := no
include            $(FRAMEWORK_DIR)/app.mk

# ---------------------------------------------------------------------------
# Custom application: rhocp_fft-opt -> links MOOSE + Marlin + our objects.
# ---------------------------------------------------------------------------
APPLICATION_DIR    := $(CURDIR)
APPLICATION_NAME   := rhocp_fft
BUILD_EXEC         := yes
GEN_REVISION       := no
include            $(FRAMEWORK_DIR)/app.mk

###############################################################################
# Additional special case targets should be added here
