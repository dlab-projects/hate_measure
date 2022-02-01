from tensorflow.keras.layers import Dense, Layer


class HateConstructLayer(Layer):
    """Maps input to multi-output categorical layers corresponding to the
    hate speech construct, as determined by survey item responses.

    These layers correspond to following survey items, which by default
    have the corresponding number of responses:
        - sentiment: 5
        - respect: 5
        - insult: 4
        - humiliate: 3
        - status: 2
        - dehumanize: 2
        - violence: 2
        - genocide: 2
        - attack-defend: 4
        - hate-speech: 2
    """
    def __init__(self, n_sentiment=5, n_respect=5, n_insult=4, n_humiliate=3,
                 n_status=2, n_dehumanize=2, n_violence=2, n_genocide=2,
                 n_attack_defend=4, n_hatespeech=2, name='construct_classification',
                 **kwargs):
        super(HateConstructLayer, self).__init__(name=name, **kwargs)
        self.n_sentiment = n_sentiment
        self.sentiment = Dense(n_sentiment, activation='softmax', name='1_sentiment')
        self.n_respect = n_respect
        self.respect = Dense(n_respect, activation='softmax', name='2_respect')
        self.n_insult = n_insult
        self.insult = Dense(n_insult, activation='softmax', name='3_insult')
        self.n_humiliate = n_humiliate
        self.humiliate = Dense(n_humiliate, activation='softmax', name='4_humiliate')
        self.n_status = n_status
        self.status = Dense(n_status, activation='softmax', name='5_status')
        self.n_dehumanize = n_dehumanize
        self.dehumanize = Dense(n_dehumanize, activation='softmax', name='6_dehumanize')
        self.n_violence = n_violence
        self.violence = Dense(n_violence, activation='softmax', name='7_violence')
        self.n_genocide = n_genocide
        self.genocide = Dense(n_genocide, activation='softmax', name='8_genocide')
        self.n_attack_defend = n_attack_defend
        self.attack_defend = Dense(n_attack_defend, activation='softmax', name='9_attack_defend')
        self.n_hatespeech = n_hatespeech
        self.hatespeech = Dense(n_hatespeech, activation='softmax', name='10_hatespeech')

    def call(self, inputs):
        outputs = [
            self.sentiment(inputs),
            self.respect(inputs),
            self.insult(inputs),
            self.humiliate(inputs),
            self.status(inputs),
            self.dehumanize(inputs),
            self.violence(inputs),
            self.genocide(inputs),
            self.attack_defend(inputs),
            self.hatespeech(inputs)
        ]
        return outputs

    def get_config(self):
        return {
            'n_sentiment': self.n_sentiment,
            'n_respect': self.n_respect,
            'n_insult': self.n_insult,
            'n_humiliate': self.n_humiliate,
            'n_status': self.n_status,
            'n_dehumanize': self.n_dehumanize,
            'n_violence': self.n_violence,
            'n_genocide': self.n_genocide,
            'n_attack_defend': self.n_attack_defend,
            'n_hatespeech': self.n_hatespeech
        }


class TargetIdentityLayer(Layer):
    """Maps input to target identity prediction.

    This layer includes the prediction of race, religion, national origin,
    gender, sexual orientation, age, disability, and an 'other' label which
    captures cases in which they might not be a clear target identity.
    """
    def __init__(self, name='target_identity_classification', **kwargs):
        super(TargetIdentityLayer, self).__init__(name=name, **kwargs)
        self.race = Dense(2, activation='softmax', name='race')
        self.religion = Dense(2, activation='softmax', name='religion')
        self.national_origin = Dense(2, activation='softmax', name='national_origin')
        self.gender = Dense(2, activation='softmax', name='gender')
        self.sexual_orientation = Dense(2, activation='softmax', name='sexual_orientation')
        self.age = Dense(2, activation='softmax', name='age')
        self.disability = Dense(2, activation='softmax', name='disability')
        self.other = Dense(2, activation='softmax', name='other')

    def call(self, inputs):
        outputs = [
            self.race(inputs),
            self.religion(inputs),
            self.national_origin(inputs),
            self.gender(inputs),
            self.sexual_orientation(inputs),
            self.age(inputs),
            self.disability(inputs),
            self.other(inputs),
        ]
        return outputs

    def get_config(self):
        return {'name': self.name}
