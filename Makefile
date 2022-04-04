CC = g++
CFLAGS = -std=c++11 -O3 -MMD 
# CFLAGS = -std=c++11 -g -MMD

HILLVALLEA_DIR := ./hv_based_MO_optimization/HillVallEA
HILLVALLEA_SRC_FILES := $(wildcard $(HILLVALLEA_DIR)/*.cpp)
HILLVALLEA_OBJ_FILES := $(patsubst $(HILLVALLEA_DIR)/%.cpp,$(HILLVALLEA_DIR)/%.o,$(HILLVALLEA_SRC_FILES))
HILLVALLEA_DEP_FILES := $(patsubst $(HILLVALLEA_DIR)/%.cpp,$(HILLVALLEA_DIR)/%.d,$(HILLVALLEA_SRC_FILES))

OPTIMAL_HYBRID_DIR := ./hv_based_MO_optimization/HillVallEA/OptimalHybrid
OPTIMAL_HYBRID_SRC_FILES := $(wildcard $(OPTIMAL_HYBRID_DIR)/*.cpp)
OPTIMAL_HYBRID_OBJ_FILES := $(patsubst $(OPTIMAL_HYBRID_DIR)/%.cpp,$(OPTIMAL_HYBRID_DIR)/%.o,$(OPTIMAL_HYBRID_SRC_FILES))
OPTIMAL_HYBRID_DEP_FILES := $(patsubst $(OPTIMAL_HYBRID_DIR)/%.cpp,$(OPTIMAL_HYBRID_DIR)/%.d,$(OPTIMAL_HYBRID_SRC_FILES))

RESOURCE_ALLOCATION_SCHEME_DIR := ./hv_based_MO_optimization/HillVallEA/resource_allocation_scheme
RESOURCE_ALLOCATION_SCHEME_SRC_FILES := $(wildcard $(RESOURCE_ALLOCATION_SCHEME_DIR)/*.cpp)
RESOURCE_ALLOCATION_SCHEME_OBJ_FILES := $(patsubst $(RESOURCE_ALLOCATION_SCHEME_DIR)/%.cpp,$(RESOURCE_ALLOCATION_SCHEME_DIR)/%.o,$(RESOURCE_ALLOCATION_SCHEME_SRC_FILES))
RESOURCE_ALLOCATION_SCHEME_DEP_FILES := $(patsubst $(RESOURCE_ALLOCATION_SCHEME_DIR)/%.cpp,$(RESOURCE_ALLOCATION_SCHEME_DIR)/%.d,$(RESOURCE_ALLOCATION_SCHEME_SRC_FILES))

UHV_SWITCH_DIR := ./hv_based_MO_optimization/HillVallEA/UHVSWITCH
UHV_SWITCH_SRC_FILES := $(wildcard $(UHV_SWITCH_DIR)/*.cpp)
UHV_SWITCH_OBJ_FILES := $(patsubst $(UHV_SWITCH_DIR)/%.cpp,$(UHV_SWITCH_DIR)/%.o,$(UHV_SWITCH_SRC_FILES))
UHV_SWITCH_DEP_FILES := $(patsubst $(UHV_SWITCH_DIR)/%.cpp,$(UHV_SWITCH_DIR)/%.d,$(UHV_SWITCH_SRC_FILES))

UHV_CLASSICAL_HYBRID_DIR := ./hv_based_MO_optimization/HillVallEA/classical_hybrid
UHV_CLASSICAL_HYBRID_SRC_FILES := $(wildcard $(UHV_CLASSICAL_HYBRID_DIR)/*.cpp)
UHV_CLASSICAL_HYBRID_OBJ_FILES := $(patsubst $(UHV_CLASSICAL_HYBRID_DIR)/%.cpp,$(UHV_CLASSICAL_HYBRID_DIR)/%.o,$(UHV_CLASSICAL_HYBRID_SRC_FILES))
UHV_CLASSICAL_HYBRID_DEP_FILES := $(patsubst $(UHV_CLASSICAL_HYBRID_DIR)/%.cpp,$(UHV_CLASSICAL_HYBRID_DIR)/%.d,$(UHV_CLASSICAL_HYBRID_SRC_FILES))

MOHILLVALLEA_DIR := ./domination_based_MO_optimization/mohillvallea
MOHILLVALLEA_SRC_FILES := $(wildcard $(MOHILLVALLEA_DIR)/*.cpp)
MOHILLVALLEA_OBJ_FILES := $(patsubst $(MOHILLVALLEA_DIR)/%.cpp,$(MOHILLVALLEA_DIR)/%.o,$(MOHILLVALLEA_SRC_FILES))
MOHILLVALLEA_DEP_FILES := $(patsubst $(MOHILLVALLEA_DIR)/%.cpp,$(MOHILLVALLEA_DIR)/%.d,$(MOHILLVALLEA_SRC_FILES))

GOMEA_DIR := ./domination_based_MO_optimization/gomea
GOMEA_SRC_FILES := $(wildcard $(GOMEA_DIR)/*.cpp)
GOMEA_OBJ_FILES := $(patsubst $(GOMEA_DIR)/%.cpp,$(GOMEA_DIR)/%.o,$(GOMEA_SRC_FILES))
GOMEA_DEP_FILES := $(patsubst $(GOMEA_DIR)/%.cpp,$(GOMEA_DIR)/%.d,$(GOMEA_SRC_FILES))

BENCHMARK_DIR :=./benchmark_functions
BENCHMARK_SRC_FILES := $(wildcard $(BENCHMARK_DIR)/*.cpp)
BENCHMARK_OBJ_FILES := $(patsubst $(BENCHMARK_DIR)/%.cpp,$(BENCHMARK_DIR)/%.o,$(BENCHMARK_SRC_FILES))
BENCHMARK_DEP_FILES := $(patsubst $(BENCHMARK_DIR)/%.cpp,$(BENCHMARK_DIR)/%.d,$(BENCHMARK_SRC_FILES))

WFG_BENCHMARK_DIR :=./benchmark_functions/wfg_Toolkit
WFG_BENCHMARK_SRC_FILES := $(wildcard $(WFG_BENCHMARK_DIR)/*.cpp)
WFG_BENCHMARK_OBJ_FILES := $(patsubst $(WFG_BENCHMARK_DIR)/%.cpp,$(WFG_BENCHMARK_DIR)/%.o,$(WFG_BENCHMARK_SRC_FILES))
WFG_BENCHMARK_DEP_FILES := $(patsubst $(WFG_BENCHMARK_DIR)/%.cpp,$(WFG_BENCHMARK_DIR)/%.d,$(WFG_BENCHMARK_SRC_FILES))

all: uhv_gomea uhv_gomea_grad uhv_hybrid_hard_switch uhv_optimal_hybrid uhv_hybrid_prototype sofomore_gomea uhv_grad uhv_adam bezea mogomea mamalgam

uhv_gomea: ./hv_based_MO_optimization/main_uhv_gomea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_gomea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

uhv_gomea_grad: ./hv_based_MO_optimization/main_uhv_gomea_grad.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_gomea_grad.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

uhv_switch: ./hv_based_MO_optimization/main_uhv_switch.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(UHV_SWITCH_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_switch.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(UHV_SWITCH_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)

#uhv_hybrid_hard_switch: ./hv_based_MO_optimization/main_uhv_gomea_adam_hard_switch.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)
#	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_gomea_adam_hard_switch.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)

uhv_optimal: ./hv_based_MO_optimization/main_uhv_optimal_hybrid.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(OPTIMAL_HYBRID_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_optimal_hybrid.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(OPTIMAL_HYBRID_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)

uhv_hybrid_naive: ./hv_based_MO_optimization/main_uhv_hybrid_naive.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(RESOURCE_ALLOCATION_SCHEME_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_hybrid_naive.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(RESOURCE_ALLOCATION_SCHEME_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)

uhv_classical_hybrid: ./hv_based_MO_optimization/main_uhv_classical_hybrid.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(UHV_CLASSICAL_HYBRID_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_classical_hybrid.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(UHV_CLASSICAL_HYBRID_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES)

sofomore_gomea: ./hv_based_MO_optimization/main_sofomore_gomea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_sofomore_gomea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

uhv_grad: ./hv_based_MO_optimization/main_uhv_grad.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_grad.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

uhv_adam: ./hv_based_MO_optimization/main_uhv_adam.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_adam.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

bezea: ./hv_based_MO_optimization/main_bezea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_bezea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

mogomea: ./domination_based_MO_optimization/main_mogomea.o  $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./domination_based_MO_optimization/main_mogomea.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

mamalgam: ./domination_based_MO_optimization/main_mamalgam.o  $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./domination_based_MO_optimization/main_mamalgam.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(HILLVALLEA_OBJ_FILES) $(HILLVALLEA_DEP_FILES) $(RESOURCE_ALLOCATION_SCHEME_OBJ_FILES) $(RESOURCE_ALLOCATION_SCHEME_DEP_FILES) $(OPTIMAL_HYBRID_OBJ_FILES) $(OPTIMAL_HYBRID_DEP_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(MOHILLVALLEA_DEP_FILES) $(GOMEA_DEP_FILES) $(BENCHMARK_DEP_FILES) $(WFG_BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_DEP_FILES) ./hv_based_MO_optimization/*.d ./hv_based_MO_optimization/*.o ./domination_based_MO_optimization/*.d ./domination_based_MO_optimization/*.o

clean_runlogs:
	rm -f *.dat
