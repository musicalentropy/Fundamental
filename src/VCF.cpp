#include "Fundamental.hpp"
#include "dsp/functions.hpp"
#include "dsp/resampler.hpp"
#include "dsp/ode.hpp"

// tanh computation using Lambert's continued fraction 
// see https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float clip(float x) {
        //return tanhf(x);
        if (x > 4.97f) {
                return 1.f;
        }
        else if (x < -4.97f) {
                return -1.f;
        }
        else {
                float x2 = x * x; 
                return x * (135135 + x2*(17325+x2*(378+x2))) / (135135 + x2*(62370+x2*(3150+28*x2)));
        }
}

struct LadderFilter {
        float omega0;
        float resonance = 1.0f;
        float state[4];
        float input;
        float lowpass;
        float highpass;

        LadderFilter() {
                reset();
                setCutoff(0.f);
        }

        void reset() {
                for (int i = 0; i < 4; i++) {
                        state[i] = 0.f;
                }
        }

        void setCutoff(float cutoff, float dt) {
                omega0 = 2.f*M_PI * cutoff;
        }

        void process(float input, float dt) {
                rack::ode::stepRK4(0.f, dt, state, 4, [&](float t, const float y[], float dydt[]) {
                        float inputc = clip(input - resonance * y[3]);
                        float yc0 = clip(y[0]);
                        float yc1 = clip(y[1]);
                        float yc2 = clip(y[2]);
                        float yc3 = clip(y[3]);

                        dydt[0] = omega0 * (inputc - yc0);
                        dydt[1] = omega0 * (yc0 - yc1);
                        dydt[2] = omega0 * (yc1 - yc2);
                        dydt[3] = omega0 * (yc2 - yc3);
                });

                lowpass = state[3];
                highpass = (input - resonance * state[3]) - 4*state[0] + 6*state[1] - 4*state[2] + state[3];
        }
};

/** Moog filter made using the TPT structure for individual LP1 filters,
    and for the feedback loop resolution in linear domain only. This
    guarantees a correct frequency response all over the audible range,
    for any cutoff frequency and resonance values, and removes any
    artefact by design when these values are modulated.

    The nonlinearities are just added "on top of the processing structure",
    hence the name "Naive NL".
*/
struct LadderFilterTPTNaiveNL {
        float G;
        float resonance = 1.0f;
        float state[4];
        float lowpass;
        float highpass;
        
        LadderFilterTPTNaiveNL() {
                reset();
                setCutoff(200.f, 44100.f);
        }

        void reset() {
                for (int i = 0; i < 4; i++) {
                        state[i] = 0.f;
                }
        }

        void setCutoff(float cutoff, float dt) {
                float g = tanf(cutoff * M_PI * dt);
                G = g / (1 + g);
        }

        void process(float input, float dt) {
                float gaincomp = 2.5f;

                resonance = clamp(resonance, 0.f, 4.0f);
                input /= gaincomp;
                
                float out = state[3] - G*(state[3] - state[2] + G*(state[2] - state[1] + G*state[1]));
                float infeed = input - resonance * out;
                                      
                float v1 = G*(infeed - state[0]);
                float y1 = clip(v1 + state[0]);
                state[0] = v1 + y1;

                float v2 = G*(y1 - state[1]);
                float y2 = clip(v2 + state[1]);
                state[1] = v2 + y2;

                float v3 = G*(y2 - state[2]);
                float y3 = clip(v3 + state[2]);
                state[2] = v3 + y3;

                float v4 = G*(y3 - state[3]);
                float y4 = v4 + state[3];
                state[3] = v4 + clip(y4);

                lowpass = gaincomp * y4;
                highpass = clip(gaincomp * ((input - resonance * y4) - 4*y1 + 6*y2 - 4*y3 + y4));
        }
};


static const int UPSAMPLE = 2;

struct VCF : Module {
        enum ParamIds {
                FREQ_PARAM,
                FINE_PARAM,
                RES_PARAM,
                FREQ_CV_PARAM,
                DRIVE_PARAM,
                NUM_PARAMS
        };
        enum InputIds {
                FREQ_INPUT,
                RES_INPUT,
                DRIVE_INPUT,
                IN_INPUT,
                NUM_INPUTS
        };
        enum OutputIds {
                LPF_OUTPUT,
                HPF_OUTPUT,
                NUM_OUTPUTS
        };

        //LadderFilter filter;
        LadderFilterTPTNaiveNL filter;
        // Upsampler<UPSAMPLE, 8> inputUpsampler;
        // Decimator<UPSAMPLE, 8> lowpassDecimator;
        // Decimator<UPSAMPLE, 8> highpassDecimator;

        VCF() : Module(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS) {}

        void onReset() override {
                filter.reset();
        }

        void step() override {
                if (!outputs[LPF_OUTPUT].active && !outputs[HPF_OUTPUT].active) {
                        outputs[LPF_OUTPUT].value = 0.f;
                        outputs[HPF_OUTPUT].value = 0.f;
                        return;
                }

                float input = inputs[IN_INPUT].value / 5.f;
                float drive = clamp(params[DRIVE_PARAM].value + inputs[DRIVE_INPUT].value / 10.f, 0.f, 1.f);
                float gain = powf(1.f + drive, 5);
                input *= gain;

                // Add -60dB noise to bootstrap self-oscillation
                input += 1e-6f * (2.f * randomUniform() - 1.f);

                // Set resonance
                float res = clamp(params[RES_PARAM].value + inputs[RES_INPUT].value / 10.f, 0.f, 1.f);
                filter.resonance = powf(res, 2) * 10.f;

                // Set cutoff frequency
                float pitch = 0.f;
                if (inputs[FREQ_INPUT].active)
                        pitch += inputs[FREQ_INPUT].value * quadraticBipolar(params[FREQ_CV_PARAM].value);
                pitch += params[FREQ_PARAM].value * 10.f - 5.f;
                pitch += quadraticBipolar(params[FINE_PARAM].value * 2.f - 1.f) * 7.f / 12.f;
                float cutoff = 261.626f * powf(2.f, pitch);
                cutoff = clamp(cutoff, 1.f, 8000.f);
                filter.setCutoff(cutoff, engineGetSampleTime());

                /*
                // Process sample
                float dt = engineGetSampleTime() / UPSAMPLE;
                float inputBuf[UPSAMPLE];
                float lowpassBuf[UPSAMPLE];
                float highpassBuf[UPSAMPLE];
                inputUpsampler.process(input, inputBuf);
                for (int i = 0; i < UPSAMPLE; i++) {
                        // Step the filter
                        filter.process(inputBuf[i], dt);
                        lowpassBuf[i] = filter.lowpass;
                        highpassBuf[i] = filter.highpass;
                }

                // Set outputs
                if (outputs[LPF_OUTPUT].active) {
                        outputs[LPF_OUTPUT].value = 5.f * lowpassDecimator.process(lowpassBuf);
                }
                if (outputs[HPF_OUTPUT].active) {
                        outputs[HPF_OUTPUT].value = 5.f * highpassDecimator.process(highpassBuf);
                }
                */
                filter.process(input, engineGetSampleTime());
                outputs[LPF_OUTPUT].value = 5.f * filter.lowpass;
                outputs[HPF_OUTPUT].value = 5.f * filter.highpass;
        }
};


struct VCFWidget : ModuleWidget {
        VCFWidget(VCF *module) : ModuleWidget(module) {
                setPanel(SVG::load(assetPlugin(plugin, "res/VCF.svg")));

                addChild(Widget::create<ScrewSilver>(Vec(15, 0)));
                addChild(Widget::create<ScrewSilver>(Vec(box.size.x - 30, 0)));
                addChild(Widget::create<ScrewSilver>(Vec(15, 365)));
                addChild(Widget::create<ScrewSilver>(Vec(box.size.x - 30, 365)));

                addParam(ParamWidget::create<RoundHugeBlackKnob>(Vec(33, 61), module, VCF::FREQ_PARAM, 0.f, 1.f, 0.5f));
                addParam(ParamWidget::create<RoundLargeBlackKnob>(Vec(12, 143), module, VCF::FINE_PARAM, 0.f, 1.f, 0.5f));
                addParam(ParamWidget::create<RoundLargeBlackKnob>(Vec(71, 143), module, VCF::RES_PARAM, 0.f, 1.f, 0.f));
                addParam(ParamWidget::create<RoundLargeBlackKnob>(Vec(12, 208), module, VCF::FREQ_CV_PARAM, -1.f, 1.f, 0.f));
                addParam(ParamWidget::create<RoundLargeBlackKnob>(Vec(71, 208), module, VCF::DRIVE_PARAM, 0.f, 1.f, 0.f));

                addInput(Port::create<PJ301MPort>(Vec(10, 276), Port::INPUT, module, VCF::FREQ_INPUT));
                addInput(Port::create<PJ301MPort>(Vec(48, 276), Port::INPUT, module, VCF::RES_INPUT));
                addInput(Port::create<PJ301MPort>(Vec(85, 276), Port::INPUT, module, VCF::DRIVE_INPUT));
                addInput(Port::create<PJ301MPort>(Vec(10, 320), Port::INPUT, module, VCF::IN_INPUT));

                addOutput(Port::create<PJ301MPort>(Vec(48, 320), Port::OUTPUT, module, VCF::LPF_OUTPUT));
                addOutput(Port::create<PJ301MPort>(Vec(85, 320), Port::OUTPUT, module, VCF::HPF_OUTPUT));
        }
};



Model *modelVCF = Model::create<VCF, VCFWidget>("Fundamental", "VCF", "VCF", FILTER_TAG);
