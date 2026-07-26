target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("../src/device/nvidia/*.cu")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_links("cudart")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("../src/ops/nvidia/*.cu")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_links("cudart")

    on_install(function (target) end)
target_end()
